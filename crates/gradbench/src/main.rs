use std::{
    env,
    ffi::OsStr,
    path::{Path, PathBuf},
    process::{exit, Command},
};

use clap::{crate_name, Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all available subcommands
    List,
    #[command(external_subcommand)]
    External(Vec<String>),
}

fn external_subcommands<'a>(exe: &'a Path, path: &'a OsStr) -> impl Iterator<Item = PathBuf> + 'a {
    let prefix = format!("{}-", exe.file_name().unwrap().to_str().unwrap());
    exe.parent()
        .map(PathBuf::from)
        .into_iter()
        .chain(env::split_paths(path))
        .flat_map(|p| p.read_dir().ok().into_iter().flatten())
        .map(|entry| entry.unwrap().path())
        .filter(move |p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            name.starts_with(&prefix) && name.chars().all(|c| c.is_alphabetic() || c == '-')
        })
}

fn main() {
    let args = Cli::parse();
    let exe_str = env::args().next().unwrap();
    let exe = Path::new(&exe_str);
    let path = env::var_os("PATH").unwrap();
    match args.command {
        Commands::List => {
            println!("Usage: gradbench <COMMAND>");
            println!();
            println!("Commands:");
            let prefix = format!("{}-", crate_name!());
            for subcmd in external_subcommands(exe, &path) {
                println!(
                    "  {}",
                    subcmd
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .strip_prefix(&prefix)
                        .unwrap()
                )
            }
        }
        Commands::External(rest) => {
            let name = format!("{}-{}", crate_name!(), rest[0]);
            exit(
                Command::new(
                    external_subcommands(exe, &path)
                        .find(|p| p.file_name().unwrap().to_str().unwrap() == name)
                        .unwrap(),
                )
                .args(rest.into_iter().skip(1))
                .status()
                .unwrap()
                .code()
                .unwrap(),
            )
        }
    }
}
