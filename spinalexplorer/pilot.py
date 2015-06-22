#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from slideio import downsample_slices
from slideio import get_slices
from slideio import PIL2array
from slideio import stack_slices
from segmentation import otsu_binarization
from registration import affine_registration
from registration import nl_registration
from registration import combine_deformations
import pvtk

rootpath = "/volatile/download/SpinalExporer_Segmentation/SCAN_5"
output_directory = "/volatile/download/SpinalExporer_Segmentation/processed"


slices = get_slices(rootpath, ".svs")
converted_slices = downsample_slices(slices, 10, "g", output_directory)

#slices = get_slices(output_directory, "^g.*.nii.gz")
masked_slices, masks, smasks, labels = otsu_binarization(
    converted_slices, 100, "o", output_directory)

#slices = get_slices(output_directory, "^og.*.nii.gz")
affine_slices = affine_registration(masked_slices, "a", output_directory)

#affine_slices = get_slices(output_directory, "^ao.*.nii.gz")
#affine_slices.insert(0, slices[0])
nl_slices = nl_registration(affine_slices, "nl", output_directory)

#slices = get_slices(output_directory, "nlao.*\d.nii.gz")
volume_file = stack_slices(nl_slices, 6 * 1e-1, "v", output_directory)
print volume_file

#volume_file = get_slices(output_directory, "vnlao.*\d.nii.gz")[0]
ren = pvtk.ren()
actor = pvtk.mesh(volume_file)
pvtk.add(ren, actor)
pvtk.show(ren)

ref_images = get_slices(output_directory, "^g.*\d.nii.gz")
affine_trfs = get_slices(output_directory, "aog.*\d.txt")
nl_fields = get_slices(output_directory, "nlaog.*-warp.nii.gz")
warped_images = combine_deformations(ref_images, affine_trfs, nl_fields,
                                     "low-", output_directory)
