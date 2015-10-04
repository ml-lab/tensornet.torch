package = "datamodule"
version = "scm-1"

source = {
   url = "git://github.com/vadimkantorov/tensornet.torch",
   tag = "master"
}

description = {
   summary = "Tensor Train layer",
   detailed = [[
	    Torch Tensor Train layer for Neural Nets (original at https://github.com/Bihaqo/TensorNet)
   ]],
   homepage = "https://github.com/vadimkantorov/tensornet.torch"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
