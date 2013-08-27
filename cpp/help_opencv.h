#ifndef HELP_OPENCV_H
#define HELP_OPENCV_H

#include <vector>

#include <opencv2/opencv.hpp>

double vec_RMS(const cv::Mat &vec);
double vec_sum(const cv::Mat &vec);

cv::Rect operator * (const cv::Rect &rect, double k);

void draw_area(cv::Mat &dst_img, std::vector<cv::Point> &area, unsigned char color);

void find_filled_area(cv::Mat &threshold_img, std::vector<cv::Point> &area,
                      const cv::Point &start_point, unsigned char pixel_value);
std::vector<std::vector<cv::Point> > find_filled_areas(cv::Mat threshold_img, unsigned char pixel_value);

void adaptive_threshold(const cv::Mat &src_img, cv::Mat &dst_img, double thresh);

double compute_skew_correction_angle(const cv::Mat &image, int threshold=300);
cv::Mat make_skew_matrix(double angle, double skew_center);

enum EdgeType
{
  ET_VERTICAL,
  ET_HORIZONTAL
};

cv::Mat compute_edge_image(const cv::Mat &image, EdgeType edge_type);

double is_rectangle(const std::vector<cv::Point> &area);

template<int ChannelsCount>
void set_white_balance(const cv::Mat &src, cv::Mat &dst,
                       const cv::Vec<unsigned char, ChannelsCount> &black,
                       const cv::Vec<unsigned char, ChannelsCount> &white)
{
  dst = cv::Mat::zeros(src.size(), src.type());
  for (int i = 0; i < src.rows; ++i)
  {
    for (int j = 0; j < src.cols; ++j)
    {
      const cv::Vec<unsigned char, ChannelsCount> &src_color = src.at<cv::Vec<unsigned char, ChannelsCount>>(i, j);
      cv::Vec<unsigned char, ChannelsCount> &dst_color = dst.at<cv::Vec<unsigned char, ChannelsCount>>(i, j);
      for (int k = 0; k < ChannelsCount; ++k)
      {
        double value = ((int)src_color[k] - (int)black[k]) * 255.0 / ((int)white[k] - (int)black[k]);
        if (value < 0.0)
        {
          dst_color[k] = 0;
        }
        else if (value > 255.0)
        {
          dst_color[k] = 255;
        }
        else
        {
          dst_color[k] = (int)value;
        }
      }
    }
  }
}

#endif // HELP_OPENCV_H
