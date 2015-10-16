TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp \
    Video.cpp \
    Utilities.cpp \
    Recognition.cpp \
    Images.cpp \
    Histograms.cpp \
    Geometric.cpp \
    Features.cpp \
    Edges.cpp \
    CameraCalibration.cpp \
    Binary.cpp

HEADERS += \
    Utilities.h

INCLUDEPATH += $$quote(D:/opencv/build/include/) \
            $$quote(D:/opencv/x86/vc11/bin/)
# vc9 = VS 2008, vc10 = VS 2010, vc11 = VS 2012, vc12 = VS 2013
LIBS += -L$$quote(D:/opencv/build/x86/vc11/lib) -lopencv_calib3d2410 -lopencv_contrib2410 -lopencv_core2410 -lopencv_features2d2410 -lopencv_flann2410 -lopencv_highgui2410 -lopencv_imgproc2410 -lopencv_legacy2410 -lopencv_ml2410 -lopencv_nonfree2410  -lopencv_objdetect2410 -lopencv_ocl2410 -lopencv_photo2410 -lopencv_stitching2410 -lopencv_superres2410 -lopencv_ts2410 -lopencv_video2410 -lopencv_videostab2410
