import qbs 1.0

Project {
    CppApplication {
        name: "ANPR"

        files: [
            "ANPR.cpp",
            "ANPR.h",
            "help_alg.h",
            "help_opencv.h",
            "help_opencv.cpp",
            "main.cpp",
            "place_number_recognizer.h",
            "place_number_recognizer.cpp"
        ]

        cpp.cxxFlags: [
            "-std=c++11"
        ]

        cpp.linkerFlags: [
            "-lopencv_core",
            "-lopencv_highgui",
            "-lopencv_imgproc",
            "-ltesseract",
            "-lboost_system",
            "-lboost_filesystem"
        ]
    }
}
