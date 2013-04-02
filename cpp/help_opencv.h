#ifndef HELP_OPENCV_H
#define HELP_OPENCV_H

#include <vector>

#include <opencv2/opencv.hpp>


double row_RMS(const cv::Mat &row);
double col_RMS(const cv::Mat &col);
double row_sum(const cv::Mat &row);
double col_sum(const cv::Mat &col);

cv::Rect operator * (const cv::Rect &rect, double k);

void draw_area(cv::Mat &dst_img, std::vector<cv::Point> &area, int color);

void walk_on_area(cv::Mat &threshold_img, std::vector<cv::Point> &area,
                  const cv::Point &point);
std::vector<std::vector<cv::Point> > find_filled_areas(cv::Mat threshold_img);

void adaptive_threshold(const cv::Mat &src_img, cv::Mat &dst_img, double thresh);


#endif // HELP_OPENCV_H
