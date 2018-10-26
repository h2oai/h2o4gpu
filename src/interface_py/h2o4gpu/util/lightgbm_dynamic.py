# pylint: skip-file

import os
import importlib.util

got_cpu_lgb = False
got_gpu_lgb = False

from h2o4gpu.util.gpu import device_count

_, ngpus_vis_global = device_count()
enable_lightgbm_import = True

if enable_lightgbm_import:
    lgb_loader = importlib.util.find_spec('lightgbm')
    lgb_found = lgb_loader is not None

    always_do_dynamic_lgb_selection = True  # False will take existing lightgbm package if exists, True will always overwrite existing
    do_dynamic_lgb_selection = True
    link_method = False  # False (default now) is to directly load from path

    if not lgb_found and do_dynamic_lgb_selection or always_do_dynamic_lgb_selection:
        numpy_loader = importlib.util.find_spec('numpy')
        found = numpy_loader is not None
        if found:
            numpy_path = os.path.dirname(numpy_loader.origin)
            dirname = "/".join(numpy_path.split("/")[:-1])
            lgb_path_gpu = os.path.join(dirname, "lightgbm_gpu")
            lgb_path_cpu = os.path.join(dirname, "lightgbm_cpu")
            lgb_path_new = os.path.join(dirname, "lightgbm")

            got_lgb = False
            expt_gpu = ""
            expt_cpu = ""
            expt_other = ""
            # This locally leads to lgb as if did import lightgbm as lgb, but also any other file that imports lgb will immediately return with lgb even though no module name "lightgbm" has a path in site-packages.
            try:
                if ngpus_vis_global > 0:
                    loader = importlib.machinery.SourceFileLoader('lightgbm',
                                                                  os.path.join(lgb_path_gpu, '__init__.py'))
                    lgb = loader.load_module()
                    print("Selected GPU version of lightgbm to import\n")
                    got_lgb = True
                    # This locally leads to lgb as if did import lightgbm as lgb, but also any other file that imports lgb will immediately return with lgb even though no module name "lightgbm" has a path in site-packages.
                    got_gpu_lgb = True
            except Exception as e:
                expt_gpu = str(e)
                pass
            if not got_lgb:
                try:
                    loader = importlib.machinery.SourceFileLoader('lightgbm',
                                                                  os.path.join(lgb_path_cpu, '__init__.py'))
                    lgb = loader.load_module()
                    if ngpus_vis_global > 0:
                        print(
                            "Selected CPU version of lightgbm to import (GPU selection failed due to %s)\n" % expt_gpu)
                    else:
                        print("Selected CPU version of lightgbm to import\n")
                    got_lgb = True
                    got_cpu_lgb = True
                except Exception as e:
                    expt_cpu = str(e)
                    pass
            if not got_lgb:
                try:
                    loader = importlib.machinery.SourceFileLoader('lightgbm',
                                                                  os.path.join(lgb_path_new, '__init__.py'))
                    lgb = loader.load_module()
                    if ngpus_vis_global > 0:
                        print(
                            "Selected non-dynamic CPU version of lightgbm to import (GPU selection failed due to %s)\n" % expt_other)
                    else:
                        print("Selected non-dynamic CPU version of lightgbm to import\n")
                    got_lgb = True
                    got_cpu_lgb = True
                except Exception as e:
                    expt_other = str(e)
                    pass
            if not got_lgb:
                print(
                    "Unable to dynamically or non-dynamically import either GPU or CPU version of lightgbm: expt_gpu=%s expt_cpu=%s expt_other=%s\n" % (
                        expt_gpu, expt_cpu, expt_other))
        else:
            print("Did not find lightgbm or numpy\n")
