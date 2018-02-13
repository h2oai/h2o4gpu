##########################################################################
# Makefile containing stderr, stdout formatting functions
##########################################################################

INFO="$$(date +%H:%M:%S)$$(tput bold)$$(tput setaf 5)$$(tput cuf 4)INFO$$(tput sgr0) **: $*"
define inform
	@echo $(INFO)$1
endef

WARN="$$(date +%H:%M:%S)$$(tput bold)$$(tput setaf 3)$$(tput cuf 1)WARNING$$(tput sgr0) **: $*"
define warn
	@echo $(WARN)$1
endef

OK  ="$$(date +%H:%M:%S)$$(tput bold)$$(tput setaf 2)$$(tput cuf 6)OK$$(tput sgr0) **: $*"
define ok
	@echo $(OK)$1
endef

ERROR = "$$(date +%H:%M:%S)$$(tput bold)$$(tput setaf 1)$$(tput cuf 2) ERROR$$(tput sgr0) **: $*"
define err
	@echo $(ERROR)$1
endef
