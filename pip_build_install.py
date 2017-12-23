#!/usr/bin/env python

'''
filename: pip_build_install.py: runs pip install in loop until 
          everything is installed.
'''

def main(argv):
    try:
        file = argv.pop(0)
    except IndexError:
        import __main__ as mod
        print("run: {} requirements_buildonly.txt [pip args]".format(mod.__file__))
    else:
        import pip
        result = 0
        with open(file,'r') as req:
            for line in req:
                word = line.strip()
                if word.startswith('#'): continue
                lib_ = pip.main(['install', word] + argv)
                result = result or lib_
        return result
    
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
