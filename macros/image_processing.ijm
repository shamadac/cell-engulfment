args = getArgument();
tokens = split(args, "|");

if (tokens.length < 2) {
    exit("Expected arguments: <nd2_dir>,<csv_output_dir>");
}

// The Python runner passes paths as a single argument separated by "|", which
// avoids ambiguity when microscopy folders contain commas or spaces.
nd2Dir = tokens[0];
csvDir = tokens[1];
processedDir = csvDir + File.separator + "processed_images";

File.makeDirectory(csvDir);
File.makeDirectory(processedDir);
setBatchMode(true);

list = getFileList(nd2Dir);
logFile = csvDir + File.separator + "processing_log.txt";
processedCount = 0;

print("ND2 directory: " + nd2Dir);
print("CSV output directory: " + csvDir);

for (i = 0; i < list.length; i++) {
    if (!endsWith(list[i], ".nd2")) {
        continue;
    }

    filename = File.getNameWithoutExtension(list[i]);
    sourcePath = nd2Dir + File.separator + list[i];

    // Bio-Formats splits the ND2 into channel windows used below. The macro is
    // retained as a compatibility path for workflows that already depend on
    // ImageJ object-table exports.
    run("Bio-Formats Importer", "open=[" + sourcePath + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT split_channels");

    yeastWindow = filename + ".nd2 - C=1";
    bacteriaWindow = filename + ".nd2 - C=2";

    print("Processing file: " + filename + ".nd2");

    // The yeast/shell channel is thresholded and counted as 3D connected
    // objects. These legacy settings produce *_scer.csv measurement tables.
    selectWindow(yeastWindow);
    run("Gaussian Blur...", "sigma=1 stack");
    setAutoThreshold("Otsu dark no-reset");
    run("Convert to Mask", "method=Otsu background=Dark calculate black");
    run("Make Binary", "method=Default background=Default black");
    run("Watershed", "stack");
    run("3D Objects Counter", "threshold=1 slice=1 min.=500 max.=159072256 objects centroids statistics summary");
    saveAs("Results", csvDir + File.separator + filename + "_scer.csv");
    close(filename + "_scer.csv");

    // Ensure transient windows are visible to ImageJ before cleanup. This helps
    // older ImageJ builds keep the correct active image after plugin calls.
    for (x = 0; x < nImages; x++) {
        selectImage(x + 1);
        run("View 100%");
    }

    // The bacterial channel uses Triangle thresholding by default because sparse
    // fluorescent objects are often poorly handled by a global Otsu threshold.
    selectWindow(bacteriaWindow);
    run("Gaussian Blur...", "sigma=1 stack");
    setAutoThreshold("Triangle dark no-reset");
    run("Convert to Mask", "method=Triangle background=Dark calculate black");
    run("Make Binary", "method=Default background=Default black");
    run("Watershed", "stack");
    run("3D Objects Counter", "threshold=1 slice=1 min.=10 max.=159072256 objects centroids statistics summary");
    saveAs("Results", csvDir + File.separator + filename + "_hflu.csv");
    close(filename + "_hflu.csv");

    // Save the binary masks as TIFF stacks for optional visual inspection.
    for (x = 0; x < nImages; x++) {
        selectImage(x + 1);
        run("View 100%");
    }

    selectWindow(yeastWindow);
    saveAs("Tiff", processedDir + File.separator + filename + "_scer.tif");
    close();

    selectWindow(bacteriaWindow);
    saveAs("Tiff", processedDir + File.separator + filename + "_hflu.tif");
    close();

    processedCount++;
    run("Close All");
}

print("Total files processed: " + processedCount);
selectWindow("Log");
saveAs("Text", logFile);
run("Quit");
