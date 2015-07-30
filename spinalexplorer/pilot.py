#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import os
import json
import shutil

from slideio import downsample_slices
from slideio import get_slices
from slideio import PIL2array
from slideio import stack_slices
from segmentation import otsu_binarization
from registration import affine_registration
from registration import nl_registration
from registration import combine_deformations
from registration import register_t1_histo
import pvtk
from reorientation import reorient_image


t1_file = "/neurospin/nsap/spinalexplorer/spinalexplorer15042015.nii.gz"
rootpath = "/neurospin/nsap/spinalexplorer"
output_directory = "/neurospin/nsap/spinalexplorer/processed"
downsample_directory = os.path.join(output_directory, "downsample")
segmentation_directory = os.path.join(output_directory, "segmentation")
inward_directory = os.path.join(output_directory, "inward")
backward_directory = os.path.join(output_directory, "backward")
horn1_directory = os.path.join(output_directory, "horn1")
horn2_directory = os.path.join(output_directory, "horn2")
nl_directory = os.path.join(output_directory, "nl")
tmp_directory = os.path.join(output_directory, "tmp")
compose_directory = os.path.join(output_directory, "compose")
start = 85
scale = 60
zsapcing = 6 * 10e-5

# 20 30e3
# 40 8000
# 60 5000
if 0:
    if not os.path.isdir(downsample_directory):
        os.mkdir(downsample_directory)
    slices = get_slices(rootpath, ".svs", sanity=True)
    converted_slices = downsample_slices(slices, scale, "g", downsample_directory)

if 0:
    if not os.path.isdir(segmentation_directory):
        os.mkdir(segmentation_directory)
    converted_slices = get_slices(downsample_directory, "^g.*.nii.gz")
    masked_slices, masks, smasks, labels = otsu_binarization(
        converted_slices, 5000, "o", segmentation_directory)

if 0:
    if not os.path.isdir(inward_directory):
        os.mkdir(inward_directory)
    if not os.path.isdir(backward_directory):
        os.mkdir(backward_directory)
    masked_slices = get_slices(segmentation_directory, "^og.*.nii.gz")
    masked_slices = [item for item in masked_slices
                    if "-1" not in item and "-2" not in item]
    affine_slices = affine_registration(
        masked_slices[start:], "a", inward_directory, thr=20)
    affine_slices = affine_registration(
        masked_slices[:start + 1][::-1], "a", backward_directory, thr=35)

if 0:
    if not os.path.isdir(horn1_directory):
        os.mkdir(horn1_directory)
    if not os.path.isdir(horn2_directory):
        os.mkdir(horn2_directory)
    masked_slices = get_slices(segmentation_directory, "^og.*.nii.gz")
    masked_slices_1 = [item for item in masked_slices if "-1" in item][::-1]
    masked_slices_2 = [item for item in masked_slices if "-2" in item][::-1]
    if len(masked_slices_1) != len(masked_slices_2):
        raise Exception("Expect equal number of horns.")
    affine_slices = affine_registration(masked_slices_1, "a", horn1_directory,
                                        thr=20)
    affine_slices = affine_registration(masked_slices_2, "a", horn2_directory,
                                        thr=20)

if 0:
    if not os.path.isdir(nl_directory):
        os.mkdir(nl_directory)
    masked_slices = get_slices(segmentation_directory, "^og.*.nii.gz")
    masked_slices = [item for item in masked_slices
                    if "-1" not in item and "-2" not in item]
    affine_slices = get_slices(inward_directory, "^aog.*.nii.gz")
    affine_slices.insert(0, masked_slices[start])
    nl_slices = nl_registration(affine_slices, "nl", nl_directory)
    affine_slices = get_slices(backward_directory, "^aog.*.nii.gz")
    affine_slices.append(masked_slices[start])
    nl_slices = nl_registration(affine_slices[::-1], "nl", nl_directory)

if 0:
    if not os.path.isdir(tmp_directory):
        os.mkdir(tmp_directory)
    nl_slices = get_slices(nl_directory, "nlao.*\d.tiff")
    for fname in nl_slices:
        shutil.copy2(
            fname, os.path.join(tmp_directory, os.path.basename(fname)))

if 0:
    nl_slices = get_slices(nl_directory, "nlao.*\d.nii.gz")
    sanity_file = os.path.join(rootpath, "sanity.json")
    sanity = json.load(open(sanity_file, "r"))
    slice_orders = sorted(sanity.keys())
    volume_file = stack_slices(
        nl_slices, zsapcing * scale, slice_orders, "rv", output_directory)
    volume_file = stack_slices(
        nl_slices, zsapcing * scale, None, "v", output_directory)

if 0:
    volume_file = get_slices(output_directory, "^vnlao.*.nii.gz")[0]
    ren = pvtk.ren()
    actor = pvtk.mesh(volume_file)
    pvtk.add(ren, actor)
    pvtk.show(ren)

if 0:
    if not os.path.isdir(compose_directory):
        os.mkdir(compose_directory)
    masked_slices = get_slices(segmentation_directory, "^og.*.nii.gz")
    masked_slices = [item for item in masked_slices
                    if "-1" not in item and "-2" not in item] 
    #get_slices(output_directory, "^g.*\d.nii.gz")
    affine_trfs = get_slices(output_directory, "aog.*\d.txt")
    nl_fields = get_slices(output_directory, "nlaog.*-warp.nii.gz")
    warped_images = combine_deformations(masked_slices, affine_trfs, nl_fields,
                                         start, "low-", compose_directory)

if 1:
    volume_file = get_slices(output_directory, "^vnlao.*.nii.gz")[0]
    reorient_image(volume_file, axes="IRP", prefix="swap_",
                   output_directory=output_directory)
    histo_file = get_slices(output_directory, "^swap_vnlao.*.nii.gz")[0]
    register_t1_histo(t1_file, histo_file, prefix="f",
                      output_directory=output_directory)
