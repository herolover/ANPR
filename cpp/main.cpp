#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <functional>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <opencv/cv.h>
#include <opencv/highgui.h>


void left_key_handler();
void right_key_handler();
void process();


enum KEY
{
  ESCAPE_KEY = 27,
  LEFT_KEY = 65363,
  RIGHT_KEY = 65361
};

const std::string folder = "../../test_img/other/";

boost::filesystem::path path(folder);
int max_img_index = std::distance(boost::filesystem::directory_iterator(path),
                                  boost::filesystem::directory_iterator());

int img_index = 0;

std::map<KEY, std::function<void ()>> key_handlers =
{
  {LEFT_KEY, left_key_handler},
  {RIGHT_KEY, right_key_handler}
};


int main()
{
  process();

  KEY key;
  while ((key = (KEY)cv::waitKey()) != ESCAPE_KEY)
  {
    if (key_handlers.count(key) == 1)
      key_handlers[key]();
  }

  return 0;
}


void left_key_handler()
{
  img_index += 1;
  if (img_index >= max_img_index)
    img_index = 0;

  process();
}


void right_key_handler()
{
  img_index -= 1;
  if (img_index < 0)
    img_index = max_img_index - 1;

  process();
}


double RMS(const cv::Mat &row)
{
  double square_sum = 0.0;
  for (int i = 0; i < row.cols; ++i)
  {
    unsigned char value = row.at<unsigned char>(0, i);
    square_sum += (double)value * value;
  }

  return sqrt(square_sum / row.cols);
}


template<class InputIt, class T> inline
T mean(InputIt first, InputIt last, T init)
{
  return std::accumulate(first, last, init) / std::distance(first, last);
}


template<class T>
std::vector<T> simple_moving_average(const std::vector<T> &values,
                                     unsigned window_size, T init)
{
  if (values.size() < window_size)
    throw std::runtime_error("SMA: points count must be greater than coverage.");

  std::vector<T> smoothed_values;
  for (auto it = values.begin(); it != values.end() - window_size + 1; ++it)
    smoothed_values.push_back(mean(it, it + window_size, init));

  return smoothed_values;
}


void process()
{
  std::cout << folder + boost::str(boost::format("%03d.jpg") % img_index) << std::endl;
  cv::Mat src_img = cv::imread(folder + boost::str(boost::format("%03d.jpg") % img_index));
  cv::Mat grayscale_img;
  cv::cvtColor(src_img, grayscale_img, CV_RGB2GRAY);

  double m[3][5] = {{-0.0, -1.0, 0.0, 1.0, 0.0},
                    {-1.0, -2.0, 0.0, 2.0, 1.0},
                    {-0.0, -1.0, 0.0, 1.0, 0.0}};
  cv::Mat edge_matrix(3, 5, CV_64FC1, m);
  cv::Mat edge_img;
  cv::filter2D(grayscale_img, edge_img, -1, edge_matrix);

  std::vector<double> rows;
  int max_row = -1;
  double max_row_value;
  for (int i = 0; i < edge_img.rows; ++i)
  {
    rows.push_back(RMS(edge_img.row(i)));

    if (max_row == -1)
    {
      max_row_value = rows[0];
      max_row = 0;
    }

    if (max_row_value < rows[i])
    {
      max_row_value = rows[i];
      max_row = i;
    }
  }

  cv::line(src_img, cv::Point(0, max_row), cv::Point(src_img.cols, max_row),
           cv::Scalar(255), 2);

  std::ofstream file;
  file.open("rows", std::ios_base::trunc);
  for (auto &row: rows)
    file << row << std::endl;
  file.close();

  file.open("row", std::ios_base::trunc);
  for (int i = 0; i < edge_img.cols; ++i)
    file << (int)edge_img.at<unsigned char>(max_row, i) << std::endl;
  file.close();

  cv::imshow("image", src_img);
}
