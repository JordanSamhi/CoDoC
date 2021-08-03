#!/bin/bash

. lib/lib.sh --source-only

print_info "Building CoDoClaSS..."
sleep 1
mvn clean install
check_return $? "Something went wrong while building CoDoClaSS with Maven." "CoDoClaSS built."
