#ifndef HELP_OPENCV_H
#define HELP_OPENCV_H

#include <vector>

#include <opencv2/opencv.hpp>


double vec_RMS(const cv::Mat &vec);
double vec_sum(const cv::Mat &vec);

cv::Rect operator * (const cv::Rect &rect, double k);

void draw_area(cv::Mat &dst_img, std::vector<cv::Point> &area, unsigned char color);

std::vector<std::vector<cv::Point> > find_filled_areas(cv::Mat threshold_img, unsigned char pixel_value);

void adaptive_threshold(const cv::Mat &src_img, cv::Mat &dst_img, double thresh);

double compute_skew_correction_angle(const cv::Mat &image);
cv::Mat make_skew_matrix(double angle, double skew_center);

cv::Mat convert_to_grayscale_and_remove_noise(const cv::Mat &image);

enum EdgeType
{
  ET_VERTICAL,
  ET_HORIZONTAL
};

cv::Mat compute_edge_image(const cv::Mat &image, EdgeType edge_type);


#endif // HELP_OPENCV_H
