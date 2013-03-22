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
    unsigned char value = row.at<unsigned char>(0, i);
    square_sum += (double)value * value;
  }

  return sqrt(square_sum / row.cols);
}

double col_RMS(const cv::Mat &col)
{
  double square_sum = 0.0;
  for (int i = 0; i < col.rows; ++i)
  {
    unsigned char value = col.at<unsigned char>(i, 0);
    square_sum += (double)value * value;
  }

  return sqrt(square_sum / col.rows);
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


template<class InputIt, class T>
std::vector<InputIt> find_local_extremums(InputIt first, InputIt last,
                                          T global, T threshold)
{
  std::vector<InputIt> local_extremums;

  while (first < last)
  {
    InputIt loc_ext = find(first, last, global, threshold);
    local_extremums.push_back(loc_ext);

    first = loc_ext + 20;
  }

  return local_extremums;
}


template<class T>
std::vector<T> compute_derivative(const std::vector<T> &values, int window_size)
{
  std::vector<double> values_der;
  for (unsigned i = 0; i < values.size() - window_size; ++i)
  {
    double sum = 0.0;
    for (unsigned j = i; j < i + window_size; ++j)
      sum += values[j + 1] - values[j];

    values_der.push_back(sum);
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
  std::vector<double> cols_der;
};


template<class InputIt>
std::vector<std::pair<int, int> > compute_local_pairs(InputIt first,
                                                      InputIt last,
                                                      int front_offset,
                                                      int back_offset,
                                                      double threshold)
{
  auto gmax_elem = std::max_element(first + front_offset, last - back_offset);
  auto gmin_elem = std::min_element(first + front_offset, last - back_offset);

  auto loc_max = find_local_extremums(first + front_offset,
                                      last - back_offset,
                                      *gmax_elem, threshold);
  auto loc_min = find_local_extremums(loc_max.front(),
                                      last - back_offset,
                                      *gmin_elem, threshold);

  std::vector<std::pair<int, int>> pairs;

  for (unsigned i = 0; i < loc_max.size() - 1; ++i)
  {
    auto max_elem = loc_max[i];
    auto next_max_elem = loc_max[i + 1];
    auto min_elem = std::find_if(loc_min.begin(), loc_min.end(),
                                 [max_elem, next_max_elem](const InputIt &point)
    {
      return point > max_elem && point < next_max_elem;
    });

    if (min_elem != loc_min.end())
    {
      std::pair<int, int> pair;
      pair.first = std::distance(first, max_elem);
      pair.second = std::distance(first, *min_elem);

      pairs.push_back(pair);
    }
  }

  return pairs;
}


void process()
{
  std::cout << folder + boost::str(boost::format("%03d.jpg") % img_index) << std::endl;
  cv::Mat src_img = cv::imread(folder + boost::str(boost::format("%03d.jpg") % img_index));

  cv::Mat grayscale_img;
  cv::cvtColor(src_img, grayscale_img, CV_RGB2GRAY);

  cv::Mat denoised_img;
//  cv::fastNlMeansDenoising(grayscale_img, denoised_img);
  denoised_img = grayscale_img;

  double m[3][5] = {{-0.2, -1.0, 0.0, 1.0, 0.2},
                    {-0.5, -2.0, 0.0, 2.0, 0.5},
                    {-0.2, -1.0, 0.0, 1.0, 0.2}};
  cv::Mat edge_matrix(3, 5, CV_64FC1, m);
  cv::Mat vert_edge_img;
  cv::filter2D(denoised_img, vert_edge_img, -1, edge_matrix);
  cv::Mat hor_edge_img;
  cv::filter2D(denoised_img, hor_edge_img, -1, edge_matrix.t());

  cv::Mat thresh_vert_edge_img;
  adap_threshold(vert_edge_img, thresh_vert_edge_img, 120);
  cv::Mat thresh_hor_edge_img;
  adap_threshold(hor_edge_img, thresh_hor_edge_img, 120);

  std::vector<double> rows;
  for (int i = 0; i < thresh_vert_edge_img.rows; ++i)
    rows.push_back(row_RMS(thresh_vert_edge_img.row(i)));

  const int img_top_offset = src_img.rows * 1 / 3;
  const int img_bottom_offset = src_img.rows * 1 / 9;

  const int window_size = 10;
  std::vector<double> rows_der = compute_derivative(rows, window_size);

  auto pairs = compute_local_pairs(rows_der.begin(), rows_der.end(),
                                   img_top_offset, img_bottom_offset, 0.5);

  std::vector<Area> areas;
  for (auto &pair: pairs)
  {
    Area area;
    pair.first += window_size;
    pair.second += window_size;
    area.rect = cv::Rect(0, pair.first, src_img.cols, pair.second - pair.first);
    areas.push_back(area);
  }

  for (auto &area: areas)
  {
    int min_line = area.rect.y - 10;
    int max_line = area.rect.y + area.rect.height + 10;

    if (min_line < 0)
      min_line = 0;
    if (max_line > thresh_hor_edge_img.rows - 1)
      max_line = thresh_hor_edge_img.rows - 1;

    cv::Mat rows = thresh_hor_edge_img.rowRange(min_line, max_line);
    for (int i = 0; i < rows.cols; ++i)
      area.cols.push_back(col_RMS(rows.col(i)));

    area.cols = simple_moving_average(area.cols, 10, 0.0);

    area.cols_der = compute_derivative(area.cols, 20);

    auto area_pairs = compute_local_pairs(area.cols_der.begin(),
                                          area.cols_der.end(), 0, 0, 0.5);
    auto broadest_pair = std::max_element(area_pairs.begin(),
                                          area_pairs.end(),
                                          [](const std::pair<int, int> &a,
                                             const std::pair<int, int> &b)
    {
      return a.second - a.first < b.second - b.first;
    });

    // yes, it's magic
    broadest_pair->first += 23;
    broadest_pair->second += 23;

    area.rect.x = broadest_pair->first;
    area.rect.width = broadest_pair->second - broadest_pair->first;
  }

  {
    save_to_file("rows", rows.begin(), rows.end());
    save_to_file("rows_der", rows_der.begin(), rows_der.end());

    for (unsigned i = 0; i < areas.size(); ++i)
    {
      save_to_file(boost::str(boost::format("cols_%1%") % i),
                   areas[i].cols.begin(), areas[i].cols.end());
      save_to_file(boost::str(boost::format("cols_der_%1%") % i),
                   areas[i].cols_der.begin(), areas[i].cols_der.end());
    }
  }

  for (auto &area: areas)
    cv::rectangle(src_img, area.rect, cv::Scalar(255), 2);

  cv::imshow("src_img", src_img);
  cv::imshow("thresh_vert_edge_img", thresh_vert_edge_img);
  cv::imshow("thresh_hor_edge_img", thresh_hor_edge_img);
}
