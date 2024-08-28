use std::{fs, path::Path};

use crate::graph::Uri;

fn builtin(path: &Path) -> Result<&'static str, ()> {
    let name = match path.to_str() {
        Some(string) => string.strip_suffix(".adroit").unwrap(),
        None => {
            eprintln!("module name is not valid Unicode: {}", path.display());
            return Err(());
        }
    };
    match name {
        "array" => Ok(include_str!("modules/array.adroit")),
        "autodiff" => Ok(include_str!("modules/autodiff.adroit")),
        "math" => Ok(include_str!("modules/math.adroit")),
        _ => {
            eprintln!("builtin module does not exist: {name}");
            Err(())
        }
    }
}

pub fn fetch(stdlib: &Uri, uri: &Uri) -> Result<String, String> {
    let uri_str = uri.as_str();
    let path = uri
        .to_file_path()
        .map_err(|()| format!("not a local file: {uri_str}"))?;
    let stdlib_path = stdlib.to_file_path().unwrap();
    match path.strip_prefix(&stdlib_path) {
        Ok(relative) => {
            let text = builtin(relative)
                .map_err(|()| format!("not a standard library module: {}", relative.display()))?;
            let parent = path.parent().unwrap();
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to make directory {}: {err}", parent.display()))?;
            fs::write(&path, text)
                .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
            Ok(text.to_owned())
        }
        Err(_) => fs::read_to_string(path).map_err(|err| format!("error reading {uri_str}: {err}")),
    }
}
