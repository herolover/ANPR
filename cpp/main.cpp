#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <functional>
#include <algorithm>
#include <utility>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <opencv2/opencv.hpp>


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


double row_RMS(const cv::Mat &row)
{
  double square_sum = 0.0;
  for (int i = 0; i < row.cols; ++i)
  {
    double value = row.at<unsigned char>(0, i) / 255.0;
    square_sum += (double)value * value;
  }

  return sqrt(square_sum / row.cols);
}

double col_RMS(const cv::Mat &col)
{
  double square_sum = 0.0;
  for (int i = 0; i < col.rows; ++i)
  {
    double value = col.at<unsigned char>(i, 0) / 255.0;
    square_sum += (double)value * value;
  }

  return sqrt(square_sum / col.rows);
}

double row_sum(const cv::Mat &row)
{
  double sum = 0.0;
  for (int i = 0; i < row.cols; ++i)
  {
    double value = row.at<unsigned char>(0, i) / 255.0;
    sum += value;
  }

  return sum;
}


double col_sum(const cv::Mat &col)
{
  double sum = 0.0;
  for (int i = 0; i < col.rows; ++i)
  {
    double value = col.at<unsigned char>(i, 0) / 255.0;
    sum += value;
  }

  return sum;
}


template<class InputIt, class T> inline
T compute_mean(InputIt first, InputIt last, T init)
{
  return std::accumulate(first, last, init) / std::distance(first, last);
}


template<class InputIt, class T>
std::pair<InputIt, InputIt> get_window_elements(const std::vector<T> &values,
                                                unsigned window_size,
                                                InputIt element)
{
  InputIt first = element - window_size / 2;
  InputIt last = element + window_size / 2 + 1;

  if (first < values.begin())
    first = values.begin();
  if (last > values.end())
    last = values.end();

  return std::make_pair(first, last);
}


template<class T>
std::vector<T> simple_moving_average(const std::vector<T> &values,
                                     unsigned window_size, T init)
{
  if (values.size() < window_size)
    throw std::runtime_error("SMA: points count must be greater than coverage.");

  std::vector<T> smoothed_values;
  for (auto element = values.begin(); element != values.end(); ++element)
  {
    auto window = get_window_elements(values, window_size, element);
    smoothed_values.push_back(compute_mean(window.first, window.second, init));
  }

  return smoothed_values;
}


template<class T>
std::vector<T> exp_smooth(const std::vector<T> &values, double alpha)
{
  std::vector<T> smoothed_values;
  smoothed_values.push_back(values.front());
  for (auto element = values.begin() + 1; element != values.end(); ++element)
    smoothed_values.push_back(smoothed_values.back() + alpha
                              * ((*element) - smoothed_values.back()));

  return smoothed_values;
}



template<class InputIt>
InputIt find_local_maximum(InputIt first, InputIt last, double threshold,
                           const std::vector<std::pair<InputIt, InputIt>> &ignored_ranges)
{
  bool is_inited = false;
  InputIt local_maximum = last;
  for (; first != last; ++first)
  {
    bool ignore_value = false;

    if (*first < threshold)
      ignore_value = true;

    for (auto &range: ignored_ranges)
      if (first > range.first && first < range.second)
      {
        ignore_value = true;
        break;
      }

    if (!ignore_value)
      if (!is_inited || *first > *local_maximum)
      {
        local_maximum = first;
        is_inited = true;
      }
  }

  return local_maximum;
}


template<class InputIt>
std::pair<InputIt, InputIt> get_threshold_bound(InputIt first, InputIt last,
                                                InputIt element, double threshold)
{
  InputIt left = element - 1;
  for (; left >= first; --left)
    if (*left < threshold)
      break;

  InputIt right = element + 1;
  for (; right < last; ++right)
    if (*right < threshold)
      break;

  return std::make_pair(left, right);
}


template<class InputIt>
std::vector<std::pair<InputIt, InputIt> > find_local_pairs(InputIt first,
                                                           InputIt last,
                                                           double threshold)
{
  auto global_maximum = std::max_element(first, last);
//  auto global_minimum = std::min_element(first, last);
  auto mean = compute_mean(first, last, 0.0);
  double abs_threshold = (*global_maximum - mean) * threshold + mean;
  std::cout << abs_threshold << std::endl;

  std::vector<std::pair<InputIt, InputIt> > local_pairs;
  std::vector<std::pair<InputIt, InputIt> > ignored_ranges;

  while (true)
  {
    InputIt local_maximum = find_local_maximum(first, last, abs_threshold,
                                               ignored_ranges);

    if (local_maximum != last)
    {
      auto bound = get_threshold_bound(first, last, local_maximum, abs_threshold);

      ignored_ranges.push_back(bound);
      if (bound.first != first && bound.second != last)
        local_pairs.push_back(bound);
    }
    else
      break;
  }

  return local_pairs;
}


template<class T>
std::vector<T> compute_derivative(const std::vector<T> &values, int window_size,
                                  T init)
{
  std::vector<double> values_der;

  for (auto element = values.begin(); element != values.end(); ++element)
  {
    auto window = get_window_elements(values, window_size, element);

    T sum = init;
    for (auto it = window.first + 1; it != window.second; ++it)
      sum += *it - *(it - 1);

    values_der.push_back(sum);// / std::distance(window.first, window.second));
  }

  return values_der;
}


void adap_threshold(const cv::Mat &src_img, cv::Mat &dst_img, double thresh)
{
  src_img.copyTo(dst_img);
  for (int i = 1; i < dst_img.cols - 1; ++i)
    for (int j = 1; j < dst_img.rows - 1; ++j)
    {
      double sum = 0.0;

      sum += dst_img.at<unsigned char>(j - 1, i - 1);
      sum += dst_img.at<unsigned char>(j - 1, i - 0);
      sum += dst_img.at<unsigned char>(j - 1, i + 1);
      sum += dst_img.at<unsigned char>(j - 0, i + 1);
      sum += dst_img.at<unsigned char>(j - 0, i - 0) * 8.0;
      sum += dst_img.at<unsigned char>(j - 0, i - 1);
      sum += dst_img.at<unsigned char>(j + 1, i - 1);
      sum += dst_img.at<unsigned char>(j + 1, i - 0);
      sum += dst_img.at<unsigned char>(j + 1, i + 1);

      if (sum / 9.0 < thresh)
        dst_img.at<unsigned char>(j, i) = 0;
    }
}


template<class InputIt>
void save_to_file(const std::string &filename, InputIt first, InputIt last)
{
  std::ofstream file(filename, std::ios_base::trunc);
  for (; first != last; ++first)
    file << *first << std::endl;
  file.close();
}


struct Area
{
  cv::Rect rect;
  std::vector<double> cols;

  cv::Mat src_img;
  cv::Mat grayscale_img;
  cv::Mat hor_img;
  cv::Mat vert_img;
};


void process()
{
  std::cout << folder + boost::str(boost::format("%03d.jpg") % img_index) << std::endl;
  cv::Mat src_img = cv::imread(folder + boost::str(boost::format("%03d.jpg") % img_index));

  cv::Mat grayscale_img;
  cv::cvtColor(src_img, grayscale_img, CV_RGB2GRAY);

  cv::Mat denoised_img;
//  cv::fastNlMeansDenoising(grayscale_img, denoised_img);
  denoised_img = grayscale_img;

  double m[3][3] = {{-1.0, 0.0, 1.0},
                    {-2.0, 0.0, 2.0},
                    {-1.0, 0.0, 1.0}};
  cv::Mat edge_matrix(3, 3, CV_64FC1, m);
  cv::Mat vert_edge_img;
  cv::filter2D(denoised_img, vert_edge_img, -1, edge_matrix);
  cv::Mat hor_edge_img;
  cv::filter2D(denoised_img, hor_edge_img, -1, edge_matrix.t());

  cv::Mat thresh_vert_edge_img;
  adap_threshold(vert_edge_img, thresh_vert_edge_img, 150);
  cv::Mat thresh_hor_edge_img;
  adap_threshold(hor_edge_img, thresh_hor_edge_img, 150);

  std::vector<double> rows;
  for (int i = 0; i < thresh_vert_edge_img.rows; ++i)
    rows.push_back(row_RMS(thresh_vert_edge_img.row(i)));

  rows = simple_moving_average(rows, 10, 0.0);

  save_to_file("rows", rows.begin(), rows.end());

  const int img_top_offset = src_img.rows * 1 / 4;
  const int img_bottom_offset = src_img.rows * 1 / 9;
  auto row_pairs = find_local_pairs(rows.begin() + img_top_offset,
                                    rows.end() - img_bottom_offset,
                                    0.4);

  std::vector<Area> areas;
  for (unsigned i = 0; i < row_pairs.size(); ++i)
  {
    int top_bound = std::distance(rows.begin(), row_pairs[i].first) - 5;
    int bottom_bound = std::distance(rows.begin(), row_pairs[i].second) + 5;

    std::cout << top_bound << " " << bottom_bound << std::endl;

    thresh_vert_edge_img = thresh_hor_edge_img;

    Area area;
    cv::Mat rows = thresh_vert_edge_img.rowRange(top_bound, bottom_bound);
    for (int i = 0; i < rows.cols; ++i)
      area.cols.push_back(col_RMS(rows.col(i)));

    area.cols = simple_moving_average(area.cols, 50, 0.0);

    save_to_file(boost::str(boost::format("cols_%1%") % i),
                 area.cols.begin(), area.cols.end());

    const int img_offset_left = src_img.cols / 8;
    const int img_offset_right = src_img.cols / 8;
    auto col_pairs = find_local_pairs(area.cols.begin() + img_offset_left,
                                      area.cols.end() - img_offset_right,
                                      0.05);


    for (auto &col_pair: col_pairs)
    {
      int left_bound = std::distance(area.cols.begin(), col_pair.first) - 5;
      int right_bound = std::distance(area.cols.begin(), col_pair.second) + 5;

      area.rect.x = left_bound;
      area.rect.y = top_bound;
      area.rect.height = bottom_bound - top_bound;
      area.rect.width = right_bound - left_bound;

      area.src_img = src_img(area.rect);
      area.grayscale_img = grayscale_img(area.rect);
      area.hor_img = thresh_hor_edge_img(area.rect);
      area.vert_img = thresh_vert_edge_img(area.rect);

      areas.push_back(area);
    }
  }

  for (auto &area: areas)
    cv::rectangle(src_img, area.rect, cv::Scalar(255), 2);

  cv::imshow("src_img", src_img);
  if (areas.size() > 0)
    cv::imshow("plate", areas[0].src_img);

  cv::imshow("thresh_vert_edge_img", thresh_vert_edge_img);
//  cv::imshow("thresh_hor_edge_img", thresh_hor_edge_img);
}
