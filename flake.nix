{
  description = "Easily call Slang functions and integrate with PyTorch auto diff directly from Python.";

  inputs =
    {
      nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
      nixpkgs-cuda.url = "github:nixos/nixpkgs/nixos-24.11";
      flake-utils.url = "github:numtide/flake-utils";
    };

  outputs = { self, nixpkgs, nixpkgs-cuda, flake-utils }:
    with flake-utils.lib;
    eachSystem [
      system.x86_64-linux
      system.aarch64-darwin
    ]
      (system:
        let
          # Setup.
          inherit (nixpkgs) lib;
          pkgs = import nixpkgs
            {
              inherit system;
            };
          pkgs-cuda = import nixpkgs-cuda
            {
              inherit system;
              config.allowUnfree = true;
            };
          # Base packages.
          basePkgs = with pkgs; [
            # Python environment.
            python3
            uv
            # Build system.
            cmake
            ninja
            # Slangpy dependencies.
            libjpeg
            libpng
            openexr_3
            asmjit
          ];
          # Linux packages (x11 and cuda required).
          linuxPkgs = with pkgs; [
            # CUDA toolkit.
            pkgs-cuda.cudatoolkit
            # Graphics libraries.
            vulkan-loader
            libGL.dev
            # X11 related libraries.
            xorg.libX11.dev
            xorg.libXi.dev
            xorg.libXrandr.dev
            xorg.libXinerama.dev
            xorg.libXcursor.dev
          ];
        in
        {
          devShells.default = pkgs.mkShell
            {
              buildInputs = basePkgs ++ (lib.optional pkgs.stdenv.isLinux linuxPkgs);
              shellHook = ''
                # Create the virtual environment if it doesn't exist
                if [ -d .venv ]; then
                  # Activate the virtual environment
                  source .venv/bin/activate
                  # Add .venv/bin to PATH
                  export PATH=$PWD/.venv/bin:$PATH
                else
                  echo "Environment not initialized."
                fi
              '';
              # Setup LD_LIBRARY_PATH on Linux.
              LD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (
                lib.makeLibraryPath (
                  pkgs.pythonManylinuxPackages.manylinux1 ++ [
                    "/run/opengl-driver"
                    pkgs.vulkan-loader
                    pkgs-cuda.cudatoolkit
                  ]
                )
              );
              # Setup CUDA_PATH on Linux.
              CUDA_PATH = lib.optionalString pkgs.stdenv.isLinux pkgs-cuda.cudatoolkit;
            };
        }
      );
}
