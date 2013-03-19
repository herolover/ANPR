#!/usr/bin/python2

import os

index = 0
for filename in os.listdir("."):
    if filename != "rename.py":
	os.rename(filename, "%03d.jpg" % index)
	index += 1