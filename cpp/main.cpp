#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <functional>
#include <algorithm>

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

const std::string folder = "../../test_img/my/";

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


template<class InputIt, class T> inline
InputIt find(InputIt first, InputIt last, T global, T threshold)
{
  return std::find_if(first, last,
                      [global, threshold](T value)
  {
    return value / global > threshold;
  });
}


void process()
{
  std::cout << folder + boost::str(boost::format("%03d.jpg") % img_index) << std::endl;
  cv::Mat src_img = cv::imread(folder + boost::str(boost::format("%03d.jpg") % img_index));
  cv::Mat grayscale_img;
  cv::cvtColor(src_img, grayscale_img, CV_RGB2GRAY);

  double m[3][5] = {{-0.2, -1.0, 0.0, 1.0, 0.2},
                    {-0.5, -2.0, 0.0, 2.0, 0.5},
                    {-0.2, -1.0, 0.0, 1.0, 0.2}};
  cv::Mat edge_matrix(3, 5, CV_64FC1, m);
  cv::Mat vert_edge_img;
  cv::filter2D(grayscale_img, vert_edge_img, -1, edge_matrix);
  cv::Mat hor_edge_img;
  cv::filter2D(grayscale_img, hor_edge_img, -1, edge_matrix.t());

  std::vector<double> rows;
  for (int i = 0; i < vert_edge_img.rows; ++i)
    rows.push_back(RMS(vert_edge_img.row(i)));

  const int img_top_offset = src_img.rows * 1 / 3;
  const int img_bottom_offset = src_img.rows * 1 / 9;

  const unsigned window_size = 10;
  std::vector<double> rows_der;
  for (unsigned i = 0; i < rows.size() - window_size; ++i)
  {
    double sum = 0.0;
    for (unsigned j = i; j < i + window_size; ++j)
      sum += rows[j + 1] - rows[j];

    rows_der.push_back(sum);
  }

  auto gmax_elem = std::max_element(rows_der.begin() + img_top_offset,
                                    rows_der.end() - img_bottom_offset);

  const double threshold = 0.75;
  std::vector<cv::Rect> rects;

  auto max_elem = rows_der.begin() + img_top_offset;
  auto min_elem = rows_der.begin() + img_top_offset;
  while (true)
  {
    max_elem = find(min_elem, rows_der.end(), *gmax_elem, threshold);
    auto next_max_elem = find(max_elem + 20, rows_der.end(), *gmax_elem, threshold);
    min_elem = std::min_element(max_elem, next_max_elem);

    if (max_elem != rows_der.end() && min_elem != rows_der.end())
    {
      int max_elem_index = std::distance(rows_der.begin(), max_elem) + window_size * 3 / 4;
      int min_elem_index = std::distance(rows_der.begin(), min_elem) + window_size * 3 / 4;

      rects.push_back(cv::Rect(0, max_elem_index,
                               src_img.cols,
                               min_elem_index - max_elem_index));
    }
    else
      break;
  }

  {
    std::ofstream file;
    file.open("rows", std::ios_base::trunc);
    for (auto &row: rows)
      file << row << std::endl;
    file.close();

    file.open("rows_der", std::ios_base::trunc);
    for (auto &row: rows_der)
      file << row << std::endl;
    file.close();
  }

  for (auto &rect: rects)
    cv::rectangle(src_img, rect, cv::Scalar(255), 2);

  cv::imshow("src_img", src_img);
  cv::imshow("vert_edge_img", vert_edge_img);
  cv::imshow("hor_edge_img", hor_edge_img);
}
