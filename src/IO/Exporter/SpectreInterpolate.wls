#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
spectreExporterLib=FindLibrary["/Users/nilsvu/Projects/spectre/build-Default-Release/lib/libBundledExporter.dylib"]


(* ::Input:: *)
spectreInterpolateImpl=ForeignFunctionLoad[
spectreExporterLib,
"spectre_interpolate_to_points",
{
"RawPointer"::["CDouble"],(* double* out*)
"RawPointer"::["RawPointer"::["UnsignedInteger8"]],(* const char** volume_files*)
"CSizeT",(* size_t num_volume_files*)
"RawPointer"::["UnsignedInteger8"],(* const char* subfile_name*)
"CInt",(* int observation_step*)
"RawPointer"::["RawPointer"::["UnsignedInteger8"]],(* const char** tensor_components*)
"CSizeT",(* size_t num_tensor_components*)
"RawPointer"::["CDouble"],(* const double* target_points*)
"CSizeT",(* size_t num_dimensions*)
"CSizeT",(* size_t num_points*)
"CUnsignedChar",(* bool extrapolate_into_excisions*)
"CUnsignedChar",(* bool error_on_missing_points*)
"CInt"  (* int num_threads*)
}->"Void"
]


(* ::Input:: *)
SpectreInterpolate[
volumeFiles_List,(*List[String]*)
subfileName_String,
observationStep_Integer,
tensorComponents_List,(*List[String]*)
targetPoints_?MatrixQ,(*dim x numPoints*)
extrapolateIntoExcisions_:False,
errorOnMissingPoints_:True,
numThreads_:0
]:=Module[
{
numVolumeFiles=Length[volumeFiles],
numTensorComponents=Length[tensorComponents],
volumeFilesPtrs,volumeFilesPtr,
tensorComponentsPtrs,tensorComponentsPtr,
targetPointsFlattened,targetPointsPtr,
outPtr,rawOut
},
{dims,numPts}=Dimensions[targetPoints];
(*1. Build char** for volume_files*)
volumeFilesPtrs=RawMemoryExport/@volumeFiles; (* each: pointer to C string*)
volumeFilesPtr=RawMemoryExport[volumeFilesPtrs,"RawPointer"::["UnsignedInteger8"]];
(*2. Build char** for tensor_components*)
tensorComponentsPtrs=RawMemoryExport/@tensorComponents;
tensorComponentsPtr=RawMemoryExport[tensorComponentsPtrs,"RawPointer"::["UnsignedInteger8"]];
(*3. Target points as flat double array, dim-major: length=dims*numPts*)
targetPointsFlattened=Flatten[targetPoints];
targetPointsPtr=RawMemoryExport[targetPointsFlattened,"CDouble"];
(*4. Allocate output buffer: numTensorComponents x numPts doubles*)
outPtr=RawMemoryAllocate["CDouble",numTensorComponents*numPts];
(*5. Call the C function*)
spectreInterpolateImpl[outPtr,volumeFilesPtr,numVolumeFiles,subfileName,observationStep,tensorComponentsPtr,numTensorComponents,targetPointsPtr,dims,numPts,If[TrueQ[extrapolateIntoExcisions],1,0],If[TrueQ[errorOnMissingPoints],1,0],numThreads];
(*6. Read back result and reshape to (numTensorComponents x numPts)*)
rawOut=RawMemoryImport[outPtr,{"NumericArray",numTensorComponents*numPts}];
ArrayReshape[rawOut,{numTensorComponents,numPts}]
];


(* ::Input:: *)
volumeFiles={"/Users/nilsvu/Projects/spectre/build-Default-Release/test_ssf/ScalarSelfForceVolume0.h5"};
subfileName="VolumeData";
observationStep=-1;
tensorComponents={"Re(MMode)"};
rstar=Range[-10,300,0.5];
costheta=0.;
XY={Table[rstar],Table[costheta,{Length[rstar]}]};
result=SpectreInterpolate[volumeFiles,subfileName,observationStep,tensorComponents,XY];
plot=ListPlot[result[[1]],
    AxesOrigin->{Min[rstar],Automatic},
    DataRange->{Min[rstar],Max[rstar]},
    PlotRange->All,
    AxesLabel->{"rstar","Re(MMode)"}
]
Export["test.png",plot]
