#!/bin/bash

. lib/lib.sh --source-only

print_info "Building CoBaClaSS..."
sleep 1
mvn clean install
check_return $? "Something went wrong while building CoBaClaSS with Maven." "CoBaClaSS built."
