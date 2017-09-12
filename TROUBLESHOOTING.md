# H2O4GPU

[![Join the chat at https://gitter.im/h2oai/h2o4gpu](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/h2oai/h2o4gpu)

## Troubleshooting

#### Good idea to check if duplicate python packages installed ###

```
pip freeze

```

and pip uninstall any prior version you had and pip install the
version we tried to install.  E.g. on conda you might need to do:

```
pip uninstall numpy
pip install numpy==1.13.1 # or whatever version was attempted to be installed by the wheel
```

### After pip installing the wheel, make sure you use a fresh bash
  environment to ensure the python cache is not used. ###

