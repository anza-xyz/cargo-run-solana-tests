use {
    anyhow::{anyhow, Context},
    regex::Regex,
    solana_bpf_loader_program::{
        create_vm, serialization::serialize_parameters, syscalls::create_loader,
    },
    solana_program_runtime::{
        compute_budget::ComputeBudget,
        executor_cache::TransactionExecutorCache,
        invoke_context::{prepare_mock_invoke_context, InvokeContext},
        log_collector::LogCollector,
        sysvar_cache::SysvarCache,
    },
    solana_rbpf::{
        elf::Executable,
        static_analysis::Analysis,
        verifier::RequisiteVerifier,
        vm::{StableResult, VerifiedExecutable},
    },
    solana_sdk::{
        account::{AccountSharedData, ReadableAccount}, bpf_loader, entrypoint::SUCCESS,
        feature_set::FeatureSet, hash::Hash, pubkey::Pubkey, sysvar::rent::Rent,
        transaction_context::TransactionContext,
    },
    std::{
        borrow::Cow,
        cell::RefCell,
        env,
        ffi::OsStr,
        fs,
        io::{self, Write},
        path::{Path, PathBuf},
        process::{exit, Command, Stdio},
        rc::Rc,
        sync::Arc,
        time::Instant,
    },
    structopt::StructOpt,
};

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
        .join("v1.35")
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
        let mut objcopy_args = vec!["--remove-section"];
        objcopy_args.push(&bss);
        objcopy_args.push(&module);
        spawn(&objcopy, &objcopy_args)?;
    }

    Ok(())
}

// Execute the given test file in Solana VM.
fn run_tests(opt: Opt) -> Result<(), anyhow::Error> {
    let path = opt.file.with_extension("so");
    let loader_id = bpf_loader::id();
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

    if !path.exists() {
        return Err(anyhow!(
            "No such file or directory: {}",
            path.to_string_lossy()
        ));
    }

    remove_bss_sections(&path)?;
    let data = fs::read(&path)?;
    let program_indices = [0, 1];
    let preparation =
        prepare_mock_invoke_context(transaction_accounts, instruction_accounts, &program_indices);
    let logs = LogCollector::new_ref_with_limit(None);
    let mut transaction_context = TransactionContext::new(
        preparation.transaction_accounts,
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
        let mut invoke_context = InvokeContext::new(
            &mut transaction_context,
            Rent::default(),
            &[],
            Cow::Owned(sysvar_cache),
            Some(Rc::clone(&logs)),
            ComputeBudget {
                compute_unit_limit: i64::MAX as u64,
                heap_size: opt.heap_size,
                ..ComputeBudget::default()
            },
            Rc::new(RefCell::new(TransactionExecutorCache::default())),
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
                &preparation.instruction_accounts,
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
        )
        .unwrap();
        let loader = create_loader(
            &invoke_context.feature_set,
            &ComputeBudget::default(),
            false, // reject_deployment_of_broken_elfs
            false, // disable_deploy_of_alloc_free_syscall
            opt.trace, // debugging_features
        )
    .unwrap();
        let executable = Executable::<InvokeContext>::from_elf(&data, loader)
            .map_err(|err| format!("Executable constructor failed: {:?}", err))
            .unwrap();
        let mut verified_executable =
            VerifiedExecutable::<RequisiteVerifier, InvokeContext>::from_executable(executable)
                .map_err(|err| format!("Executable verifier failed: {:?}", err))
                .unwrap();
        #[cfg(all(not(target_os = "windows"), target_arch = "x86_64"))]
        verified_executable.jit_compile().unwrap();
        let mut vm = create_vm(
            &verified_executable,
            regions,
            account_lengths,
            &mut invoke_context,
        )
        .unwrap();
        let start_time = Instant::now();
        let (instruction_count, result) = vm.execute_program(false);
        println!(
            "Executed {} {} instructions in {:.2}s.",
            path.to_string_lossy(),
            instruction_count,
            start_time.elapsed().as_secs_f64()
        );
        if opt.trace {
            println!("Trace:");
            let trace_log = vm
                .env
                .context_object_pointer
                .trace_log_stack
                .last()
                .expect("Inconsistent trace log stack")
                .trace_log
                .as_slice();
            let analysis = Analysis::from_executable(verified_executable.get_executable()).unwrap();
            analysis
                .disassemble_trace_log(&mut std::io::stdout(), trace_log)
                .unwrap();
        }
        result
    };

    if let Ok(logs) = Rc::try_unwrap(logs) {
        for message in Vec::from(logs.into_inner()) {
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
    name = "cargo-run-sbf-tests",
    about = "Test runner for the sbf-solana-solana target"
)]
struct Opt {
    #[allow(dead_code)]
    #[structopt(long, hidden = true)]
    quiet: bool,
    /// Solana VM heap size
    #[structopt(long)]
    heap_size: Option<usize>,
    #[structopt(short)]
    trace: bool,
    #[structopt(parse(from_os_str))]
    file: PathBuf,
}

fn main() {
    solana_logger::setup();

    let mut args = env::args().collect::<Vec<_>>();
    if let Some("run-sbf-tests") = args.get(1).map(|a| a.as_str()) {
        // we're being invoked as `cargo run-sbf-tests`
        args.remove(1);
    }

    let opt = Opt::from_iter(&args);
    if let Err(e) = run_tests(opt) {
        eprintln!("error: {:#}", e);
        exit(1);
    }
}
