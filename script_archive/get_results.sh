#!/bin/bash

grep -r "Total cells" *weak*.log 
grep -r "Particle Compute" *weak*.log
grep -r "Flow Compute" *weak*.log
grep -r "Program" *weak*.log

grep -r "Total cells" *strong*.log 
grep -r "Particle Compute" *strong*.log
grep -r "Flow Compute" *strong*.log
grep -r "Program" *strong*.log
