#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import bisect
import numpy
import nibabel
from scipy import ndimage
import matplotlib.pyplot as plt

# IO import
from PIL import Image

# Opencv import
import cv2


def otsu_binarization(slices, min_blob_size=100, prefix="o",
                      output_directory=None, verbose=1):
    """ Automatic segmentation using Otsu's segmentation.

    Automatically calculates a threshold value from each slice histogram
    considering a bimodal distribution.
    Perform a connected component analysis and mask the slice with the largest
    blob.

    <process>
        <return name="masked_slices" type="List_File" desc="The compressed
            nifti masked slices."/>
        <return name="masks" type="List_File" desc="The tiff otsu binarization
            result."/>
        <return name="smasks" type="List_File" desc="The largest blob mask."/>
        <return name="labels" type="List_File" desc="The tiff connected
            components result."/>
        <input name="slices" type="List_File" desc="The slices to segment."/>
        <input name="min_blob_size" type="Int" desc="A threshold to remove
            small blobs from the connected components analysis."/>
        <input name="prefix" type="Str" desc="The segmentation results
            prefix."/>
        <input name="output_directory" type="Directory" desc="The destination
            folder."/>
        <input name="verbose" type="Int" desc="If greater than zero display
            debuging information, if greater than one display images."/>
    </process>
    """
    # Check some slices have been passed as an input
    if len(slices) == 0:
        raise ValueError("No slice has been specified.".format(
            output_directory))

    # Check that the outdir is valid
    output_directory = output_directory or os.path.dirname(slices[0])
    if not os.path.isdir(output_directory):
        raise ValueError("'{0}' is not a valid directory.".format(
            output_directory))

    # Got through each slice
    labels = []
    masks = []
    smasks = []
    masked_slices = []
    for filename in slices:

        if verbose > 0:
            print "-" * 10
            print "File:", filename
            print "Size threshold:", min_blob_size
            print "-" * 10

        # Load the slice
        image = nibabel.load(filename)
        array = image.get_data()
        affine = image.get_affine()

        # Otsu's binarization
        # blur = cv2.GaussianBlur(numpy.cast[numpy.uint8](array), (5, 5), 0)
        ret, thresh = cv2.threshold(numpy.cast[numpy.uint8](array), 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Close to remove small hole
        # kernel = numpy.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        #                      dtype=numpy.uint8)
        kernel = numpy.ones((5, 5), dtype=numpy.uint8)
        mask = ndimage.morphology.binary_closing(thresh, kernel, iterations=2)
        mask = ndimage.morphology.binary_dilation(mask, kernel, iterations=1)
        mask = fill_2d_holes(mask)
        mask = numpy.cast[numpy.uint8](mask) * 255

        # Compute connected components - filter small blob
        blob_sizes = []
        im_labels, nb_labels = ndimage.label(mask)
        keep_labels = []
        for label in range(nb_labels):
            label_indices = numpy.where(im_labels==label)
            blob_size = len(label_indices[0])
            if blob_size <  min_blob_size:
                im_labels[label_indices] = 0
            elif label != 0:
                index = bisect.bisect(blob_sizes, blob_size)
                keep_labels.insert(index, label)
                blob_sizes.insert(index, blob_size)

        # Get/apply a mask built from the two largest components
        # label = blobs[max(blobs.keys())]
        if len(keep_labels) <= 0 or len(keep_labels) > 2:
            if len(keep_labels) > 2:
                keep_labels = keep_labels[:2]
            else: 
                raise Exception(
                    "One or two connected components are expected not '{0}'.".format(
                    len(keep_labels)))
        smask = numpy.zeros(im_labels.shape, dtype=numpy.uint8)
        arrays = []
        ycenters = []
        for label in keep_labels:
            label_indices = numpy.where(im_labels==label)
            ycenter = numpy.mean(label_indices[1])
            index = bisect.bisect(ycenters, ycenter)
            ycenters.insert(index, ycenter)
            label_array = numpy.zeros(im_labels.shape, dtype=numpy.float)
            label_array[label_indices] = array[label_indices]
            arrays.insert(index, label_array)
            smask[label_indices] = 255
        array[numpy.where(smask == 0)] = 0

        if verbose > 1:
            # Callback to check mask
            numrows, numcols = array.shape
            def format_coord(x, y):
                col = int(x + 0.5)
                row = int(y + 0.5)
                if col >= 0 and col < numcols and row >= 0 and row < numrows:
                    value = mask[row, col]
                    return "x=%1.4f, y=%1.4f, masked=%i" % (x, y, value)
                else:
                    return "x=%1.4f, y=%1.4f" % (x, y)
            plt.figure()
            ax = plt.subplot(221)
            ax.format_coord = format_coord
            plt.imshow(array)
            plt.axis("off")
            ax = plt.subplot(222)
            ax.format_coord = format_coord
            plt.imshow(mask, cmap=plt.cm.gray, interpolation="nearest")
            plt.axis("off")
            ax = plt.subplot(223)
            ax.format_coord = format_coord
            plt.imshow(im_labels)
            plt.axis("off")
            ax = plt.subplot(224)
            ax.format_coord = format_coord
            plt.imshow(smask, cmap=plt.cm.gray, interpolation="nearest")
            plt.axis("off")
            plt.show()

        # Save the results
        fname = os.path.basename(filename).split(".")[0]
        outfile = os.path.join(output_directory, prefix + fname)
        nibabel.save(image, outfile + ".nii.gz")
        image = Image.fromarray(numpy.cast[numpy.uint8](array))
        image.save(outfile + ".tiff")
        image = Image.fromarray(numpy.cast[numpy.uint8](mask))
        image.save(outfile + "-mask.tiff")
        image = Image.fromarray(numpy.cast[numpy.uint8](smask))
        image.save(outfile + "-smask.tiff")
        image = Image.fromarray(numpy.cast[numpy.uint8](im_labels))
        image.save(outfile + "-labels.tiff")

        # Update outputs
        labels.append(outfile + "-labels.tiff")
        masks.append(outfile + "-mask.tiff")
        smasks.append(outfile + "-smask.tiff")
        masked_slices.append(outfile + ".nii.gz")

        # Save branches
        component_prefix = ["-1", "-2"]
        if len(arrays) == 2:
            for index, array in enumerate(arrays):
                outfile = os.path.join(
                    output_directory, prefix + fname + component_prefix[index])
                image = nibabel.Nifti1Image(array, affine)
                nibabel.save(image, outfile + ".nii.gz")
                image = Image.fromarray(numpy.cast[numpy.uint8](array))
                image.save(outfile + ".tiff")

    return masked_slices, masks, smasks, labels


def fill_2d_holes(slide):
    """ Fill 2D cavities using 6-connectivity.
    """
    # Create a cross structural element
    structure = ndimage.morphology.generate_binary_structure(2, 1)
    
    # Fill holes
    filled_slide = ndimage.binary_fill_holes(slide, structure).astype(
        slide.dtype)

    return filled_slide
