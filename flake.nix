{
  description = "AOC development";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # custom cuda package for lib and lib64 dirs for nvvm
        cuda-compat = pkgs.symlinkJoin {
          name = "cuda-compat";
          paths = [ pkgs.cudatoolkit ];
          postBuild = ''
            ln -s $out/lib $out/lib64

            if [ -d "$out/nvvm/lib" ] && [ ! -d "$out/nvvm/lib64" ]; then
              ln -s $out/nvvm/lib $out/nvvm/lib64
            fi
          '';
        };
        # python packages to install
        requirementsFile = pkgs.writeText "requirements.txt" ''
          numpy
          numba
          numba-cuda
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python315

            cuda-compat # custom cuda package

            cmake
            gcc
            gdb
            clang
            perf
            stdenv.cc.cc.lib
            zlib
          ];

          shellHook = ''
            echo "AOC development environment loading..."

            # Setup Library Paths
            export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${cuda-compat}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${cuda-compat}/nvvm/lib64:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH

            export CUDA_HOME=${cuda-compat}

            # Venv Setup
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment..."
              python -m venv .venv
            fi

            # Universal Activation (Bash/Zsh/Fish)
            # We explicitly modify PATH so the shell prioritizes the venv.
            export VIRTUAL_ENV=$PWD/.venv
            export PATH=$VIRTUAL_ENV/bin:$PATH

            # Sync Packages
            echo "Syncing pip packages..."
            pip install -r ${requirementsFile} --quiet --upgrade

            echo "Environment ready! Python: $(which python)"
          '';
        };
      }
    );
}
