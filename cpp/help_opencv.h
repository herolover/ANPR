#ifndef HELP_OPENCV_H
#define HELP_OPENCV_H

#include <vector>

#include <opencv2/opencv.hpp>

double vec_RMS(const cv::Mat &vec);
double vec_sum(const cv::Mat &vec);

cv::Rect operator * (const cv::Rect &rect, double k);

void draw_area(cv::Mat &dst_img, const std::vector<cv::Point> &area,
               unsigned char color, const cv::Point &offset=cv::Point(0, 0));

void find_filled_area(cv::Mat &threshold_img, std::vector<cv::Point> &area,
                      const cv::Point &start_point, unsigned char pixel_value);
std::vector<std::pair<std::vector<cv::Point>, cv::Rect>> find_filled_areas(const cv::Mat &threshold_img,
                                                                           unsigned char pixel_value);

void adaptive_threshold(const cv::Mat &src_img, cv::Mat &dst_img, double thresh);

double compute_skew_correction_angle(const cv::Mat &image, int threshold=300);
cv::Mat make_skew_matrix(double angle, double skew_center);

enum EdgeType
{
  ET_VERTICAL,
  ET_HORIZONTAL
};

bool is_intersects(const cv::Rect &a, const cv::Rect &b);
cv::Rect expand(const cv::Rect &a, const cv::Rect &b);

void compute_edge_image(const cv::Mat &src, cv::Mat &dst, EdgeType edge_type);

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

template<int ChannelsCount>
std::pair<cv::Vec<unsigned char, ChannelsCount>,
          cv::Vec<unsigned char, ChannelsCount> > find_min_max_in_rect_center(const cv::Mat &image)
{
  cv::Rect rect_center;
  rect_center.width = image.cols / 10;
  rect_center.height = image.rows / 10;
  rect_center.x = image.cols / 2 - rect_center.width / 2;
  rect_center.y = image.rows / 2 - rect_center.height / 2;

  cv::Mat search_image = image(rect_center);
//  cv::imshow("search_image", search_image);
  auto minmax_color = std::minmax_element(search_image.begin<cv::Vec<unsigned char, ChannelsCount>>(),
                                          search_image.end<cv::Vec<unsigned char, ChannelsCount>>(),
                                          [](const cv::Vec<unsigned char, ChannelsCount> &a,
                                             const cv::Vec<unsigned char, ChannelsCount> &b)
  {
    return cv::norm(a) < cv::norm(b);
  });

  return std::make_pair(*minmax_color.first, *minmax_color.second);
}

#endif // HELP_OPENCV_H
