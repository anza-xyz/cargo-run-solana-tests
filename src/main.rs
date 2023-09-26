use {
    anyhow::{anyhow, Context},
    regex::Regex,
    solana_bpf_loader_program::{
        create_vm, load_program_from_bytes, serialization::serialize_parameters,
        syscalls::create_program_runtime_environment_v1,
    },
    solana_program_runtime::{
        compute_budget::ComputeBudget, invoke_context::InvokeContext,
        loaded_programs::{LoadedProgramsForTxBatch, LoadProgramMetrics, LoadedProgramType},
        log_collector::LogCollector, sysvar_cache::SysvarCache,
    },
    solana_rbpf::{
        elf::Executable,
        static_analysis::Analysis,
        vm::StableResult,
    },
    solana_sdk::{
        account::{AccountSharedData, ReadableAccount}, bpf_loader_upgradeable,
        entrypoint::SUCCESS, feature_set::{self, FeatureSet}, hash::Hash, pubkey::Pubkey,
        sysvar::rent::Rent, slot_history::Slot, transaction_context::TransactionContext,
    },
    std::{
        convert::TryFrom,
        env,
        ffi::OsStr,
        fs::File,
        io::{self, Read, Seek, Write},
        path::{Path, PathBuf},
        process::{exit, Command, Stdio},
        rc::Rc,
        sync::Arc,
        time::Instant,
    },
    structopt::StructOpt,
};

// Replace with std::lazy::Lazy when stabilized.
// https://github.com/rust-lang/rust/issues/74465
struct LazyAnalysis<'a, 'b> {
    analysis: Option<Analysis<'a>>,
    executable: &'a Executable<InvokeContext<'b>>,
}

impl<'a, 'b> LazyAnalysis<'a, 'b> {
    fn new(executable: &'a Executable<InvokeContext<'b>>) -> Self {
        Self {
            analysis: None,
            executable,
        }
    }

    fn analyze(&mut self) -> &Analysis {
        if let Some(ref analysis) = self.analysis {
            return analysis;
        }
        self.analysis
            .insert(Analysis::from_executable(self.executable).unwrap())
    }
}

// Start a new process running the program and capturing its output.
fn spawn<I, S>(program: &Path, args: I) -> Result<(String, String), anyhow::Error>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let child = Command::new(program)
        .args(args)
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to execute {}", program.display()))?;

    let output = child.wait_with_output().expect("failed to wait on child");
    Ok((
        String::from_utf8(output.stdout).context("process stdout is not valid utf8")?,
        String::from_utf8(output.stderr).context("process stderr is not valid utf8")?,
    ))
}

fn extract_sections_list(output: &str) -> Vec<String> {
    let head_re = Regex::new(r"^ +\[[ 0-9]+\] (.bss[^ ]*) .*").unwrap();

    let mut result: Vec<String> = Vec::new();
    for line in output.lines() {
        let line = line.trim_end();
        if let Some(captures) = head_re.captures(line) {
            result.push(captures[1].to_string());
        }
    }

    result
}

fn llvm_home() -> Result<PathBuf, anyhow::Error> {
    if let Ok(home) = env::var("LLVM_HOME") {
        return Ok(home.into());
    }

    let home_dir = PathBuf::from(env::var("HOME").context("Can't get home directory path")?);
    Ok(home_dir
        .join(".cache")
        .join("solana")
        .join("v1.38")
        .join("platform-tools")
        .join("llvm"))
}

fn remove_bss_sections(module: &Path) -> Result<(), anyhow::Error> {
    let module = module.to_string_lossy();
    let llvm_path = llvm_home()?.join("bin");
    let readelf = llvm_path.join("llvm-readelf");
    let mut readelf_args = vec!["--section-headers"];
    readelf_args.push(&module);

    let output = spawn(&readelf, &readelf_args)?.0;
    let sections = extract_sections_list(&output);

    for bss in sections {
        let objcopy = llvm_path.join("llvm-objcopy");
        let mut objcopy_args = vec!["--strip-all", "--remove-section"];
        objcopy_args.push(&bss);
        objcopy_args.push(&module);
        spawn(&objcopy, &objcopy_args)?;
    }

    Ok(())
}

fn load_program<'a>(
    filename: &Path,
    program_id: Pubkey,
    invoke_context: &InvokeContext<'a>,
) -> Executable<InvokeContext<'a>> {
    let mut file = File::open(filename).unwrap();
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).unwrap();
    file.rewind().unwrap();
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).unwrap();
    let slot = Slot::default();
    let log_collector = invoke_context.get_log_collector();
    let loader_key = bpf_loader_upgradeable::id();
    let mut load_program_metrics = LoadProgramMetrics {
        program_id: program_id.to_string(),
        ..LoadProgramMetrics::default()
    };
    let account_size = contents.len();
    let program_runtime_environment = create_program_runtime_environment_v1(
        &invoke_context.feature_set,
        invoke_context.get_compute_budget(),
        false, /* deployment */
        false,  /* debugging_features */
    )
    .unwrap();
    // Allowing mut here, since it may be needed for jit compile, which is under a config flag
    #[allow(unused_mut)]
    let mut verified_executable = {
        let result = load_program_from_bytes(
            invoke_context
                .feature_set
                .is_active(&feature_set::delay_visibility_of_program_deployment::id()),
            log_collector,
            &mut load_program_metrics,
            &contents,
            &loader_key,
            account_size,
            slot,
            Arc::new(program_runtime_environment),
            false,
        );
        match result {
            Ok(loaded_program) => match loaded_program.program {
                LoadedProgramType::LegacyV1(program) => Ok(program),
                _ => unreachable!(),
            },
            Err(err) => Err(format!("Loading executable failed: {err:?}")),
        }
    }
    .unwrap();
    #[cfg(all(not(target_os = "windows"), target_arch = "x86_64"))]
    verified_executable.jit_compile().unwrap();
    unsafe {
        std::mem::transmute::<Executable<InvokeContext<'static>>, Executable<InvokeContext<'a>>>(
            verified_executable,
        )
    }
}

// Execute the given test file in Solana VM.
fn run_tests(opt: Opt) -> Result<(), anyhow::Error> {
    let path = opt.file.with_extension("so");
    if !path.exists() {
        return Err(anyhow!(
            "No such file or directory: {}",
            path.to_string_lossy()
        ));
    }

    remove_bss_sections(&path)?;

    let loader_id = bpf_loader_upgradeable::id();
    let program_id = Pubkey::new_unique();
    let transaction_accounts = vec![
        (
            loader_id,
            AccountSharedData::new(0, 0, &solana_sdk::native_loader::id()),
        ),
        (
            Pubkey::new_unique(),
            AccountSharedData::new(0, 0, &loader_id),
        ),
    ];
    let instruction_accounts = Vec::new();
    let program_indices = [0, 1];
    let logs = LogCollector::new_ref_with_limit(None);
    let mut transaction_context = TransactionContext::new(
        transaction_accounts,
        Some(Rent::default()),
        1,
        1,
    );
    let mut sysvar_cache = SysvarCache::default();
    sysvar_cache.fill_missing_entries(|pubkey, callback| {
        for index in 0..transaction_context.get_number_of_accounts() {
            if transaction_context
                .get_key_of_account_at_index(index)
                .unwrap()
                == pubkey
            {
                callback(
                    transaction_context
                        .get_account_at_index(index)
                        .unwrap()
                        .borrow()
                        .data(),
                );
            }
        }
    });
    let result = {
        let programs_loaded_for_tx_batch = LoadedProgramsForTxBatch::default();
        let mut programs_modified_by_tx = LoadedProgramsForTxBatch::default();
        let mut programs_updated_only_for_global_cache = LoadedProgramsForTxBatch::default();
        let mut invoke_context = InvokeContext::new(
            &mut transaction_context,
            Rent::default(),
            &sysvar_cache,
            Some(Rc::clone(&logs)),
            ComputeBudget {
                compute_unit_limit: i64::MAX as u64,
                heap_size: opt.heap_size.unwrap(),
                ..ComputeBudget::default()
            },
            &programs_loaded_for_tx_batch,
            &mut programs_modified_by_tx,
            &mut programs_updated_only_for_global_cache,
            Arc::new(FeatureSet::all_enabled()),
            Hash::default(),
            0,
            0,
        );
        let instruction_data = vec![];
        invoke_context
            .transaction_context
            .get_next_instruction_context()
            .unwrap()
            .configure(
                &program_indices,
                &instruction_accounts,
                &instruction_data,
            );
        invoke_context.push().unwrap();
        let (_parameter_bytes, regions, account_lengths) = serialize_parameters(
            invoke_context.transaction_context,
            invoke_context
                .transaction_context
                .get_current_instruction_context()
                .unwrap(),
            true, // should_cap_ix_accounts
            true, // copy_account_data
        )
        .unwrap();

        let verified_executable = load_program(path.as_path(), program_id, &invoke_context);
        create_vm!(
            vm,
            &verified_executable,
            regions,
            account_lengths,
            &mut invoke_context
        );
        let mut vm = vm.unwrap();
        let start_time = Instant::now();
        let (instruction_count, result) = vm.execute_program(&verified_executable, false);
        println!(
            "Executed {} {} instructions in {:.2}s.",
            path.to_string_lossy(),
            instruction_count,
            start_time.elapsed().as_secs_f64()
        );
        if opt.trace {
            println!("Trace:");
            let mut analysis = LazyAnalysis::new(&verified_executable);
            if let Some(Some(syscall_context)) = vm.context_object_pointer.syscall_context.last() {
                let trace = syscall_context.trace_log.as_slice();
                analysis
                    .analyze()
                    .disassemble_trace_log(&mut std::io::stdout(), trace)
                    .unwrap();
            }
        }
        result
    };

    if let Ok(logs) = Rc::try_unwrap(logs) {
        for message in logs.into_inner().into_messages() {
            let _ = io::stdout().write_all(message.replace("Program log: ", "").as_bytes());
        }
    }

    match result {
        StableResult::Ok(exit_code) => {
            if exit_code == SUCCESS {
                Ok(())
            } else {
                Err(anyhow!("exit code: {}", exit_code))
            }
        }
        StableResult::Err(e) => {
            // if false {
            //     let trace = File::create("trace.out").unwrap();
            //     let mut trace = BufWriter::new(trace);
            //     let analysis =
            //         solana_rbpf::static_analysis::Analysis::from_executable(executable.as_ref());
            //     vm.get_tracer().write(&mut trace, &analysis).unwrap();
            // }
            Err(anyhow!("{:?}", e))
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "cargo-run-solana-tests",
    about = "Test runner for the Solana Virtual Machine target"
)]
struct Opt {
    #[allow(dead_code)]
    #[structopt(long, hidden = true)]
    quiet: bool,
    /// Solana VM heap size
    #[structopt(long)]
    heap_size: Option<u32>,
    #[structopt(short)]
    trace: bool,
    #[structopt(parse(from_os_str))]
    file: PathBuf,
}

fn main() {
    solana_logger::setup();

    let mut args = env::args().collect::<Vec<_>>();
    if let Some("run-solana-tests") = args.get(1).map(|a| a.as_str()) {
        // we're being invoked as `cargo run-solana-tests`
        args.remove(1);
    }

    let opt = Opt::from_iter(&args);
    if let Err(e) = run_tests(opt) {
        eprintln!("error: {:#}", e);
        exit(1);
    }
}
