use solana_bpf_loader_program::{
    BpfError,
    create_vm,
    serialization::serialize_parameters,
    syscalls::register_syscalls,
    ThisInstructionMeter,
};
use solana_rbpf::vm::{
    Config,
    Executable
};
use solana_sdk::{
    account::AccountSharedData,
    bpf_loader,
    entrypoint::SUCCESS,
    keyed_account::KeyedAccount,
    process_instruction::{InvokeContext, MockInvokeContext},
};
use regex::Regex;
use std::{
    cell::RefCell,
    env,
    ffi::OsStr,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, exit, Stdio},
    time::Instant,
};

/**
 * Start a new process running the program and capturing its output.
 */
fn spawn<I, S>(program: &Path, args: I) -> (String, String)
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args = args.into_iter().collect::<Vec<_>>();
    print!("Running on host: {}", program.display());
    for arg in args.iter() {
        print!(" {}", arg.as_ref().to_str().unwrap_or("?"));
    }
    println!();

    let child = Command::new(program)
        .args(&args)
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap_or_else(|err| {
            eprintln!("Failed to execute {}: {}", program.display(), err);
            exit(1);
        });

    let output = child.wait_with_output().expect("failed to wait on child");
    (
        output
            .stdout
            .as_slice()
            .iter()
            .map(|&c| c as char)
            .collect::<String>(),
        output
            .stderr
            .as_slice()
            .iter()
            .map(|&c| c as char)
            .collect::<String>()
    )
}

/**
 * Run cargo test to build the test binaries and collect the list of the test modules to run.
 */
fn run_cargo_test<I, S>(args: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let cargo = PathBuf::from("cargo");
    let mut cargo_args = vec![
        "+bpf",
        "test",
        "-v",
        "--target",
        "bpfel-unknown-unknown",
    ];
    let args = args.into_iter().collect::<Vec<_>>();
    for arg in args.iter() {
        cargo_args.push(arg.as_ref().to_str().unwrap_or(""));
    }
    spawn(&cargo, &cargo_args).1
}

fn cargo_test_failed(cargo_output: &String) -> bool {
    let expected = Regex::new(r"error: test failed, to rerun pass '.+'").unwrap();
    if !expected.is_match(cargo_output) {
        eprintln!("{}", cargo_output);
        return true;
    }
    false
}

/**
 * Extract the list of binary modules built by cargo test from the command's output.
 */
fn extract_tests_list(output: &String) -> Vec<String> {
    let rust_re = Regex::new(r"^\s*Running `[^ ]*rustc .*--target bpfel-unknown-unknown.+").unwrap();
    let odir_re = Regex::new(r"^.+--out-dir ([^ ]+).+").unwrap();
    let name_re = Regex::new(r"^.+--crate-name ([^ ]+).+-C extra-filename=([^ ]+).+").unwrap();
    let mut result: Vec<String> = Vec::new();
    let lines = output.lines().collect::<Vec<_>>();
    for line in lines {
        let line = line.trim_end();
        if rust_re.is_match(line) {
            if odir_re.is_match(line) {
                let captures = odir_re.captures(line).unwrap();
                let base = captures[1].to_string();
                if name_re.is_match(line) {
                    let captures = name_re.captures(line).unwrap();
                    result.push(format!("{}/{}{}.so", base, captures[1].to_string(), captures[2].to_string()));
                }
            }
        }
    }
    result
}

fn extract_sections_list(output: &String) -> Vec<String> {
    let head_re = Regex::new(r"^ +\[[ 0-9]+\] (.bss[^ ]*) .*").unwrap();
    let mut result: Vec<String> = Vec::new();
    let lines = output.lines().collect::<Vec<_>>();
    for line in lines {
        let line = line.trim_end();
        if head_re.is_match(line) {
            let captures = head_re.captures(line).unwrap();
            result.push(format!("{}", captures[1].to_string()));
        }
    }
    result
}

fn remove_bss_sections(module: &String) {
    let home_dir = PathBuf::from(env::var("HOME").unwrap_or_else(|err| {
        eprintln!("Can't get home directory path: {}", err);
        exit(1);
    }));
    let llvm_path = home_dir
        .join(".cache")
        .join("solana")
        .join("v1.15")
        .join("bpf-tools")
        .join("llvm")
        .join("bin");
    let readelf = llvm_path.join("llvm-readelf");
    let mut readelf_args = vec!["--section-headers"];
    readelf_args.push(module);
    let output = spawn(&readelf, &readelf_args).0;
    let sections = extract_sections_list(&output);
    for bss in sections {
        let objcopy = llvm_path.join("llvm-objcopy");
        let mut objcopy_args = vec!["--remove-section"];
        objcopy_args.push(&bss);
        objcopy_args.push(module);
        spawn(&objcopy, &objcopy_args);
    }
}

fn is_executable(module: &String) -> bool {
    let home_dir = PathBuf::from(env::var("HOME").unwrap_or_else(|err| {
        eprintln!("Can't get home directory path: {}", err);
        exit(1);
    }));
    let llvm_path = home_dir
        .join(".cache")
        .join("solana")
        .join("v1.15")
        .join("bpf-tools")
        .join("llvm")
        .join("bin");
    let readelf = llvm_path.join("llvm-readelf");
    let mut readelf_args = vec!["--section-headers"];
    readelf_args.push(module);
    let output = spawn(&readelf, &readelf_args).0;
    let head_re = Regex::new(r"^ +\[[ 0-9]+\] (.text[^ ]*) .*").unwrap();
    let lines = output.lines().collect::<Vec<_>>();
    for line in lines {
        let line = line.trim_end();
        if head_re.is_match(line) {
            return true
        }
    }
    false
}

/**
 * Execute the test binary modules in RBPF.
 */
fn run_tests(tests: &Vec<String>) -> bool {
    let mut failed = false;
    let config = Config {
        max_call_depth: 100,
        enable_instruction_tracing: false,
        ..Config::default()
    };
    let loader_id = bpf_loader::id();
    let key = solana_sdk::pubkey::new_rand();
    let program_id = solana_sdk::pubkey::new_rand();
    let mut account = RefCell::new(AccountSharedData::default());
    for program in tests {
        eprintln!("Considering {}", program);
        let path = PathBuf::from(&program);
        if !path.exists() || !is_executable(&program) {
            eprintln!("  Skipping...");
            continue;
        }
        remove_bss_sections(&program);
        let mut file = File::open(path).unwrap();
        let mut data = vec![];
        file.read_to_end(&mut data).unwrap();
        let accounts = vec![KeyedAccount::new(&key, false, &mut account)];
        let parameters = serialize_parameters(&bpf_loader::id(), &program_id, &accounts, &[]).unwrap();
        // Make new context for every test module, otherwise log messages from previous runs accumulate.
        // DO NOT move outside the loop.
        let mut invoke_context = MockInvokeContext::new(accounts);
        let logger = invoke_context.logger.clone();
        let compute_meter = invoke_context.get_compute_meter();
        let mut instruction_meter = ThisInstructionMeter { compute_meter };
        let syscall_registry = register_syscalls(&mut invoke_context).unwrap();
        let mut executable = <dyn Executable<BpfError, ThisInstructionMeter>>::from_elf(&data, None, config, syscall_registry).unwrap();
        executable.jit_compile().unwrap();
        let mut parameters = parameters.clone();
        let mut vm = create_vm(&loader_id, executable.as_ref(), parameters.as_slice_mut(), &mut invoke_context).unwrap();
        let start_time = Instant::now();
        let result = vm.execute_program_jit(&mut instruction_meter);
        let instruction_count = vm.get_total_instruction_count();
        println!("Executed {} {} instructions in {:.2}s.", program, instruction_count, start_time.elapsed().as_secs_f64());
        for s in logger.log.borrow_mut().iter() {
            println!("{}", s);
        }
        match result {
            Err(e) => {
                println!("FAILURE {}\n", e);
                failed = true;
                // FIX: commented-out traces
                #[cfg(target_arch = "no_real")]
                if false {
                    let trace = File::create("trace.out").unwrap();
                    let mut trace = BufWriter::new(trace);
                    let analysis = solana_rbpf::static_analysis::Analysis::from_executable(executable.as_ref());
                    vm.get_tracer().write(&mut trace, &analysis).unwrap();
                }
            },
            Ok(v) => {
                if v == SUCCESS {
                    println!("SUCCESS\n");
                } else {
                    println!("Exit code {}\n", v);
                }
            }
        };
    }
    failed
}

fn main() {
    solana_logger::setup();
    let mut args = env::args().collect::<Vec<_>>();
    if let Some(arg1) = args.get(1) {
        if arg1 == "run-bpf-tests" {
            args.remove(1);
        }
    }
    if let Some(_) = args.get(0) {
        args.remove(0);
    }
    let cargo_output = run_cargo_test(&args);
    println!("cargo test output:\n{}", cargo_output);
    if cargo_test_failed(&cargo_output) {
        exit(1);
    }
    let tests_list = extract_tests_list(&cargo_output);
    println!("Found the following tests");
    for test in &tests_list {
        println!("  {}", test);
    }
    let failed = run_tests(&tests_list);
    if failed {
        exit(1);
    }
}
