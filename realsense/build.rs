fn main() {
    // Some Homebrew bottles ship a .pc file with a Linux libdir. Find the actual
    // dylib under the package prefix without changing the system installation.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let output = std::process::Command::new("pkg-config")
            .args(["--variable=prefix", "realsense2"])
            .output()
            .expect("pkg-config is required");
        if output.status.success() {
            let prefix = String::from_utf8(output.stdout).unwrap();
            let lib = std::path::Path::new(prefix.trim()).join("lib");
            if lib.join("librealsense2.dylib").exists() {
                println!("cargo:rustc-link-search=native={}", lib.display());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
            }
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
}
