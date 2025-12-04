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
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python313
            python313Packages.pip
            python313Packages.numpy
            python313Packages.numba

            cudatoolkit

            cmake
            gcc
            gdb
            clang
            perf
          ];

          shellHook = ''
            echo "AOC development environment loadeda"

            export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/nvvm/lib:$LD_LIBRARY_PATH
          '';
        };
      }
    );
}
