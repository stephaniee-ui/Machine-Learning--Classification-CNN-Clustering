#!/bin/sh
prefix="CNN3"
dir=$(realpath ~ctsai314/scratch)
alias cp='cp -v'

cp ${dir}/${prefix}_best.ckpt .
cp ${dir}/${prefix}_best.sum .
cp ${dir}/${prefix}_loss.pkl .
cp ${dir}/${prefix}_log.txt .

cp ${dir}/train_80p.csv ~/scratch/
cp ${dir}/test_10p.csv ~/scratch/
cp ${dir}/validation_10p.csv ~/scratch/

echo "Checksum for the model here: "
md5sum ${prefix}_best.ckpt

echo "Checksum saved in the origin: "
cat ${prefix}_best.sum
echo ""
