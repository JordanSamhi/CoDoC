#!/bin/bash

. lib/lib.sh --source-only

print_info "Building CoDoC..."
sleep 1
mvn clean install
check_return $? "Something went wrong while building CoDoC with Maven." "CoDoC built."
