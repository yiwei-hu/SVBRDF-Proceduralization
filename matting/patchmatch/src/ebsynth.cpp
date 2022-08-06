// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy
// and modify this file as you see fit.

#include "ebsynth.h"
#include "ebsynth_cpu.h"
#include "ebsynth_cuda.h"

#include <cmath>
#include <cstdio>

EBSYNTH_API
void ebsynthRun(int    ebsynthBackend,
                int    numStyleChannels,
                int    numGuideChannels,
                int    sourceWidth,
                int    sourceHeight,
                void*  sourceStyleData,
                void*  sourceGuideData,
                int    targetWidth,
                int    targetHeight,
                void*  targetGuideData,
                void*  targetModulationData,
                float* styleWeights,
                float* guideWeights,
                float  uniformityWeight,
                int    patchSize,
                int    voteMode,
                int    numPyramidLevels,
                int*   numSearchVoteItersPerLevel,
                int*   numPatchMatchItersPerLevel,
                int*   stopThresholdPerLevel,
                int    extraPass3x3,
                void*  outputNnfData,
                void*  outputImageData)
{
    void (*backendDispatch)(int, int, int, int, void*, void*, int, int, void*, void*, float*, float*, float, int, int, int, int*, int*, int*, int, void*, void*) = 0;

    if (ebsynthBackend == EBSYNTH_BACKEND_CPU) {
        backendDispatch = ebsynthRunCpu;
    } else if (ebsynthBackend == EBSYNTH_BACKEND_CUDA) {
        backendDispatch = ebsynthRunCuda;
    } else if (ebsynthBackend == EBSYNTH_BACKEND_AUTO) {
        backendDispatch = ebsynthBackendAvailableCuda() ? ebsynthRunCuda : ebsynthRunCpu;
    }

    if (backendDispatch != 0) {
        backendDispatch(numStyleChannels,
                        numGuideChannels,
                        sourceWidth,
                        sourceHeight,
                        sourceStyleData,
                        sourceGuideData,
                        targetWidth,
                        targetHeight,
                        targetGuideData,
                        targetModulationData,
                        styleWeights,
                        guideWeights,
                        uniformityWeight,
                        patchSize,
                        voteMode,
                        numPyramidLevels,
                        numSearchVoteItersPerLevel,
                        numPatchMatchItersPerLevel,
                        stopThresholdPerLevel,
                        extraPass3x3,
                        outputNnfData,
                        outputImageData);
    }
}

EBSYNTH_API
int ebsynthBackendAvailable(int ebsynthBackend)
{
    if (ebsynthBackend == EBSYNTH_BACKEND_CPU) {
        return ebsynthBackendAvailableCpu();
    } else if (ebsynthBackend == EBSYNTH_BACKEND_CUDA) {
        return ebsynthBackendAvailableCuda();
    } else if (ebsynthBackend == EBSYNTH_BACKEND_AUTO) {
        return ebsynthBackendAvailableCpu() || ebsynthBackendAvailableCuda();
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "jzq.h"

template <typename FUNC>
bool tryToParseArg(const std::vector<std::string>& args, int* inout_argi, const char* name, bool* out_fail, FUNC handler)
{
    int&  argi = *inout_argi;
    bool& fail = *out_fail;

    if (argi < 0 || argi >= args.size()) {
        fail = true;
        return false;
    }

    if (args[argi] == name) {
        argi++;
        fail = !handler();
        return true;
    }

    fail = false;
    return false;
}

bool tryToParseIntArg(const std::vector<std::string>& args, int* inout_argi, const char* name, int* out_value, bool* out_fail)
{
    return tryToParseArg(args, inout_argi, name, out_fail, [&] {
        int& argi = *inout_argi;
        if (argi < args.size()) {
            const std::string& arg = args[argi];
            try {
                std::size_t pos = 0;
                *out_value      = std::stoi(arg, &pos);
                if (pos != arg.size()) {
                    printf("error: bad %s argument '%s'\n", name, arg.c_str());
                    return false;
                }
                return true;
            } catch (...) {
                printf("error: bad %s argument '%s'\n", name, arg.c_str());
                return false;
            }
        }
        printf("error: missing argument for the %s option\n", name);
        return false;
    });
}

bool tryToParseFloatArg(const std::vector<std::string>& args, int* inout_argi, const char* name, float* out_value, bool* out_fail)
{
    return tryToParseArg(args, inout_argi, name, out_fail, [&] {
        int& argi = *inout_argi;
        if (argi < args.size()) {
            const std::string& arg = args[argi];
            try {
                std::size_t pos = 0;
                *out_value      = std::stof(arg, &pos);
                if (pos != arg.size()) {
                    printf("error: bad %s argument '%s'\n", name, arg.c_str());
                    return false;
                }
                return true;
            } catch (...) {
                printf("error: bad %s argument '%s'\n", name, args[argi].c_str());
                return false;
            }
        }
        printf("error: missing argument for the %s option\n", name);
        return false;
    });
}

bool tryToParseStringArg(const std::vector<std::string>& args, int* inout_argi, const char* name, std::string* out_value, bool* out_fail)
{
    return tryToParseArg(args, inout_argi, name, out_fail, [&] {
        int& argi = *inout_argi;
        if (argi < args.size()) {
            *out_value = args[argi];
            return true;
        }
        printf("error: missing argument for the %s option\n", name);
        return false;
    });
}

bool tryToParseStringPairArg(const std::vector<std::string>& args, int* inout_argi, const char* name, std::pair<std::string, std::string>* out_value, bool* out_fail)
{
    return tryToParseArg(args, inout_argi, name, out_fail, [&] {
        int& argi = *inout_argi;
        if ((argi + 1) < args.size()) {
            *out_value = std::make_pair(args[argi], args[argi + 1]);
            argi++;
            return true;
        }
        printf("error: missing argument for the %s option\n", name);
        return false;
    });
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

unsigned char* tryLoad(const std::string& fileName, int* width, int* height)
{
    unsigned char* data = stbi_load(fileName.c_str(), width, height, NULL, 4);
    if (data == NULL) {
        printf("error: failed to load '%s'\n", fileName.c_str());
        printf("%s\n", stbi_failure_reason());
        exit(1);
    }
    return data;
}

int evalNumChannels(const unsigned char* data, const int numPixels)
{
    bool isGray   = true;
    bool hasAlpha = false;

    for (int xy = 0; xy < numPixels; xy++) {
        const unsigned char r = data[xy * 4 + 0];
        const unsigned char g = data[xy * 4 + 1];
        const unsigned char b = data[xy * 4 + 2];
        const unsigned char a = data[xy * 4 + 3];

        if (!(r == g && g == b)) {
            isGray = false;
        }
        if (a < 255) {
            hasAlpha = true;
        }
    }

    const int numChannels = (isGray ? 1 : 3) + (hasAlpha ? 1 : 0);

    return numChannels;
}

V2i pyramidLevelSize(const V2i& sizeBase, const int level)
{
    return V2i(V2f(sizeBase) * std::pow(2.0f, -float(level)));
}

std::string backendToString(const int ebsynthBackend)
{
    if (ebsynthBackend == EBSYNTH_BACKEND_CPU) {
        return "cpu";
    } else if (ebsynthBackend == EBSYNTH_BACKEND_CUDA) {
        return "cuda";
    } else if (ebsynthBackend == EBSYNTH_BACKEND_AUTO) {
        return "auto";
    }
    return "unknown";
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("usage: %s [options]\n", argv[0]);
        printf("\n");
        printf("options:\n");
        printf("  -svbrdf_dir <directory>\n");
        printf("  -output_dir <directory>\n");
        printf("  -mask <mask.png>\n");
        printf("  -weight <value>\n");
        printf("  -uniformity <value>\n");
        printf("  -patchsize <size>\n");
        printf("  -pyramidlevels <number>\n");
        printf("  -searchvoteiters <number>\n");
        printf("  -patchmatchiters <number>\n");
        printf("  -stopthreshold <value>\n");
        printf("  -extrapass3x3\n");
        printf("  -backend [cpu|cuda]\n");
        printf("\n");
        return 1;
    }

    struct Image {
        int            width;
        int            height;
        int            channels;
        unsigned char* data;
    };

    std::vector<Image> svbrdfs;
    std::string        svbrdf_dir;
    std::string        output_dir;
    std::string        albedoFileName;
    std::string        normalFileName;
    std::string        roughnessFileName;
    std::string        maskFileName;

    float uniformityWeight   = 3500;
    float maskWeight         = 1.0;
    int   patchSize          = 5;
    int   numPyramidLevels   = -1;
    int   numSearchVoteIters = 6;
    int   numPatchMatchIters = 4;
    int   stopThreshold      = 1;
    int   extraPass3x3       = 0;
    int   backend            = ebsynthBackendAvailable(EBSYNTH_BACKEND_CUDA) ? EBSYNTH_BACKEND_CUDA : EBSYNTH_BACKEND_CPU;

    {
        std::vector<std::string> args(argc);
        for (int i = 0; i < argc; i++) {
            args[i] = argv[i];
        }

        bool fail = false;
        int  argi = 1;

        while (argi < argc && !fail) {
            float       weight;
            std::string backendName;

            if (tryToParseStringArg(args, &argi, "-svbrdf_dir", &svbrdf_dir, &fail)) {
                argi++;
            } else if (tryToParseStringArg(args, &argi, "-output_dir", &output_dir, &fail)) {
                argi++;
            } else if (tryToParseStringArg(args, &argi, "-mask", &maskFileName, &fail)) {
                argi++;
            } else if (tryToParseFloatArg(args, &argi, "-weight", &maskWeight, &fail)) {
                argi++;
            } else if (tryToParseFloatArg(args, &argi, "-uniformity", &uniformityWeight, &fail)) {
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-patchsize", &patchSize, &fail)) {
                if (patchSize < 3) {
                    printf("error: patchsize is too small!\n");
                    return 1;
                }
                if (patchSize % 2 == 0) {
                    printf("error: patchsize must be an odd number!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-pyramidlevels", &numPyramidLevels, &fail)) {
                if (numPyramidLevels < 1) {
                    printf("error: bad argument for -pyramidlevels!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-searchvoteiters", &numSearchVoteIters, &fail)) {
                if (numSearchVoteIters < 0) {
                    printf("error: bad argument for -searchvoteiters!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-patchmatchiters", &numPatchMatchIters, &fail)) {
                if (numPatchMatchIters < 0) {
                    printf("error: bad argument for -patchmatchiters!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseIntArg(args, &argi, "-stopthreshold", &stopThreshold, &fail)) {
                if (stopThreshold < 0) {
                    printf("error: bad argument for -stopthreshold!\n");
                    return 1;
                }
                argi++;
            } else if (tryToParseStringArg(args, &argi, "-backend", &backendName, &fail)) {
                if (backendName == "cpu") {
                    backend = EBSYNTH_BACKEND_CPU;
                } else if (backendName == "cuda") {
                    backend = EBSYNTH_BACKEND_CUDA;
                } else {
                    printf("error: unrecognized backend '%s'\n", backendName.c_str());
                    return 1;
                }

                if (!ebsynthBackendAvailable(backend)) {
                    printf("error: the %s backend is not available!\n", backendToString(backend).c_str());
                    return 1;
                }

                argi++;
            } else if (argi < args.size() && args[argi] == "-extrapass3x3") {
                extraPass3x3 = 1;
                argi++;
            } else {
                printf("error: unrecognized option '%s'\n", args[argi].c_str());
                fail = true;
            }
        }

        if (output_dir.empty()) {
            output_dir = svbrdf_dir;
        }

        auto it = svbrdf_dir.end() - 1;
        if (*it == '/' || *it == '\\') {
            svbrdf_dir.erase(it);
        }
        albedoFileName    = svbrdf_dir + "/albedo.png";
        normalFileName    = svbrdf_dir + "/normal.png";
        roughnessFileName = svbrdf_dir + "/roughness.png";

        if (fail) {
            return 1;
        }
    }

    int   numSVBRDFChannelsTotal = 0;
    Image albedo;
    albedo.data     = tryLoad(albedoFileName, &albedo.width, &albedo.height);
    albedo.channels = evalNumChannels(albedo.data, albedo.width * albedo.height);
    numSVBRDFChannelsTotal += albedo.channels;
    svbrdfs.push_back(albedo);
#ifndef EBSYNTH_ALBEDO_ONLY
    Image normal;
    normal.data     = tryLoad(normalFileName, &normal.width, &normal.height);
    normal.channels = evalNumChannels(normal.data, normal.width * normal.height);
    numSVBRDFChannelsTotal += normal.channels;
    svbrdfs.push_back(normal);

    if (albedo.width != normal.width || albedo.height != normal.height) {
        printf("error: shape mismatch, albedo shape is %dx%dx%d, normal shape is %dx%dx%d\n", albedo.width, albedo.height, albedo.channels, normal.width, normal.height, normal.channels);
        return 1;
    }

    Image roughness;
    roughness.data     = tryLoad(roughnessFileName, &roughness.width, &roughness.height);
    roughness.channels = evalNumChannels(roughness.data, roughness.width * roughness.height);
    numSVBRDFChannelsTotal += roughness.channels;
    svbrdfs.push_back(roughness);

    if (albedo.width != roughness.width || albedo.height != roughness.height) {
        printf("error: shape mismatch, albedo shape is %dx%dx%d, normal shape is %dx%dx%d\n", albedo.width, albedo.height, albedo.channels, roughness.width, roughness.height, roughness.channels);
        return 1;
    }
#endif
    std::vector<unsigned char> sourceSVBRDF(albedo.width * albedo.height * numSVBRDFChannelsTotal);
    for (int xy = 0; xy < albedo.width * albedo.height; xy++) {
        int c = 0;
        for (size_t i = 0; i < svbrdfs.size(); i++) {
            const int channels = svbrdfs[i].channels;

            if (channels > 0) {
                sourceSVBRDF[xy * numSVBRDFChannelsTotal + c + 0] = svbrdfs[i].data[xy * 4 + 0];
            }
            if (channels == 2) {
                sourceSVBRDF[xy * numSVBRDFChannelsTotal + c + 1] = svbrdfs[i].data[xy * 4 + 3];
            } else if (channels > 1) {
                sourceSVBRDF[xy * numSVBRDFChannelsTotal + c + 1] = svbrdfs[i].data[xy * 4 + 1];
            }
            if (channels > 2) {
                sourceSVBRDF[xy * numSVBRDFChannelsTotal + c + 2] = svbrdfs[i].data[xy * 4 + 2];
            }
            if (channels > 3) {
                sourceSVBRDF[xy * numSVBRDFChannelsTotal + c + 3] = svbrdfs[i].data[xy * 4 + 3];
            }

            c += channels;
        }
    }

    Image mask;
    mask.data     = tryLoad(maskFileName, &mask.width, &mask.height);
    mask.channels = evalNumChannels(mask.data, mask.width * mask.height);

    if (albedo.width != mask.width || albedo.height != mask.height) {
        printf("error: shape mismatch, source shape is %dx%dx%d, mask shape is %dx%dx%d\n", albedo.width, albedo.height, albedo.channels, mask.width, mask.height, mask.channels);
        return 1;
    }
    if (numSVBRDFChannelsTotal > EBSYNTH_MAX_STYLE_CHANNELS) {
        printf("error: too many style channels (%d), maximum number is %d\n", numSVBRDFChannelsTotal, EBSYNTH_MAX_STYLE_CHANNELS);
        return 1;
    }
    if (mask.channels > EBSYNTH_MAX_GUIDE_CHANNELS) {
        printf("error: too many guide channels (%d), maximum number is %d\n", mask.channels, EBSYNTH_MAX_GUIDE_CHANNELS);
        return 1;
    }

    std::vector<unsigned char> sourceMask(mask.width * mask.height * mask.channels);
    for (int xy = 0; xy < mask.width * mask.height; xy++) {
        if (mask.channels > 0) {
            sourceMask[xy * mask.channels + 0] = mask.data[xy * 4 + 0];
        }
        if (mask.channels == 2) {
            sourceMask[xy * mask.channels + 1] = mask.data[xy * 4 + 3];
        } else if (mask.channels > 1) {
            sourceMask[xy * mask.channels + 1] = mask.data[xy * 4 + 1];
        }
        if (mask.channels > 2) {
            sourceMask[xy * mask.channels + 2] = mask.data[xy * 4 + 2];
        }
        if (mask.channels > 3) {
            sourceMask[xy * mask.channels + 3] = mask.data[xy * 4 + 3];
        }
    }
    std::vector<unsigned char> targetMask;
    for (auto& pixel : sourceMask) {
        targetMask.push_back(255 - pixel);
    }

    std::vector<float> svbrdfWeights(numSVBRDFChannelsTotal);
    for (int i = 0; i < numSVBRDFChannelsTotal; i++) {
        svbrdfWeights[i] = 1.0 / float(numSVBRDFChannelsTotal);
    }

    std::vector<float> maskWeights(mask.channels);
    for (int i = 0; i < mask.channels; i++) {
        maskWeights[i] = maskWeight / float(mask.channels);
    }

    int maxPyramidLevels = 0;
    for (int level = 32; level >= 0; level--) {
        if (min(pyramidLevelSize(V2i(albedo.width, albedo.height), level)) >= (2 * patchSize + 1)) {
            maxPyramidLevels = level + 1;
            break;
        }
    }

    if (numPyramidLevels == -1) {
        numPyramidLevels = maxPyramidLevels;
    }
    numPyramidLevels = std::min(numPyramidLevels, maxPyramidLevels);

    std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
    std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
    std::vector<int> stopThresholdPerLevel(numPyramidLevels);
    for (int i = 0; i < numPyramidLevels; i++) {
        numSearchVoteItersPerLevel[i] = numSearchVoteIters;
        numPatchMatchItersPerLevel[i] = numPatchMatchIters;
        stopThresholdPerLevel[i]      = stopThreshold;
    }

    std::vector<unsigned char> output(albedo.width * albedo.height * numSVBRDFChannelsTotal);

    printf("uniformity: %.1f\n", uniformityWeight);
    printf("weight: %.1f\n", maskWeight);
    printf("patchsize: %d\n", patchSize);
    printf("pyramidlevels: %d\n", numPyramidLevels);
    printf("searchvoteiters: %d\n", numSearchVoteIters);
    printf("patchmatchiters: %d\n", numPatchMatchIters);
    printf("stopthreshold: %d\n", stopThreshold);
    printf("extrapass3x3: %s\n", extraPass3x3 != 0 ? "yes" : "no");
    printf("backend: %s\n", backendToString(backend).c_str());

    ebsynthRun(backend,
               numSVBRDFChannelsTotal,
               mask.channels,
               albedo.width,
               albedo.height,
               sourceSVBRDF.data(),
               sourceMask.data(),
               mask.width,
               mask.height,
               targetMask.data(),
               NULL,
               svbrdfWeights.data(),
               maskWeights.data(),
               uniformityWeight,
               patchSize,
               EBSYNTH_VOTEMODE_WEIGHTED,
               numPyramidLevels,
               numSearchVoteItersPerLevel.data(),
               numPatchMatchItersPerLevel.data(),
               stopThresholdPerLevel.data(),
               extraPass3x3,
               NULL,
               output.data());

    auto it = output_dir.end() - 1;
    if (*it == '/' || *it == '\\') {
        output_dir.erase(it);
    }
    std::string albedoOutput    = output_dir + "/albedo_inpainted.png";
    std::string normalOutput    = output_dir + "/normal_inpainted.png";
    std::string roughnessOutput = output_dir + "/roughness_inpainted.png";

#ifndef EBSYNTH_ALBEDO_ONLY
    std::vector<unsigned char> albedoData;
    std::vector<unsigned char> normalData;
    std::vector<unsigned char> roughnessData;
    for (int xy = 0; xy < albedo.width * albedo.height; xy++) {
        for (int i = 0; i < numSVBRDFChannelsTotal; i++) {
            if (i < albedo.channels) {
                albedoData.push_back(output[xy * numSVBRDFChannelsTotal + i]);
            } else if (i < albedo.channels + normal.channels) {
                normalData.push_back(output[xy * numSVBRDFChannelsTotal + i]);
            } else {
                roughnessData.push_back(output[xy * numSVBRDFChannelsTotal + i]);
            }
        }
    }

    stbi_write_png(albedoOutput.c_str(), albedo.width, albedo.height, albedo.channels, albedoData.data(), albedo.channels * albedo.width);
    stbi_write_png(normalOutput.c_str(), normal.width, normal.height, normal.channels, normalData.data(), normal.channels * normal.width);
    stbi_write_png(roughnessOutput.c_str(), roughness.width, roughness.height, roughness.channels, roughnessData.data(), roughness.channels * roughness.width);

    printf("albedo was written to %s\n", albedoOutput.c_str());
    printf("normal was written to %s\n", normalOutput.c_str());
    printf("roughness was written to %s\n", roughnessOutput.c_str());
#else
    stbi_write_png(albedoOutput.c_str(), albedo.width, albedo.height, albedo.channels, output.data(), albedo.channels * albedo.width);

    printf("albedo was written to %s\n", albedoOutput.c_str());
#endif

    for (size_t i = 0; i < svbrdfs.size(); i++) {
        stbi_image_free(svbrdfs[i].data);
    }
    stbi_image_free(mask.data);
    return 0;
}
