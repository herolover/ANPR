#ifndef HELP_ALG_H
#define HELP_ALG_H

#include <vector>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <fstream>


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


template<class InputIt>
std::pair<InputIt, InputIt> get_threshold_bound(InputIt first, InputIt last,
                                                InputIt element, double threshold)
{
  InputIt left = element - 1;
  for (; left > first; --left)
    if (*left < threshold)
      break;

  InputIt right = element + 1;
  for (; right < last; ++right)
    if (*right < threshold)
      break;

  return std::make_pair(left, right);
}


template<class InputIt, class T> inline
T compute_mean(InputIt first, InputIt last, T init)
{
  return std::accumulate(first, last, init) / std::distance(first, last);
}


template<class T> inline
T sqr(T value)
{
  return value * value;
}


template<class InputIt, class T> inline
T compute_RMS(InputIt first, InputIt last, T init)
{
  T square_sum = init;
  for (InputIt it = first; it != last; ++it)
    square_sum += sqr(*it);

  return sqrt(square_sum / std::distance(first, last));
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
      if (first >= range.first && first < range.second)
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
std::vector<std::pair<InputIt, InputIt> > find_local_pairs(InputIt first,
                                                           InputIt last,
                                                           double threshold)
{
  auto global_maximum = std::max_element(first, last);
  auto global_minimum = std::min_element(first, last);
//  auto mean = compute_mean(first, last, 0.0);
  double abs_threshold = (*global_maximum - *global_minimum) * threshold + *global_minimum;

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
      if (bound.first != local_maximum && bound.second != local_maximum)
        local_pairs.push_back(bound);
    }
    else
      break;
  }

  return local_pairs;
}


template<class InputIt>
void save_to_file(const std::string &filename, InputIt first, InputIt last)
{
  std::ofstream file(filename, std::ios_base::trunc);
  for (; first != last; ++first)
    file << *first << std::endl;
  file.close();
}


#endif // HELP_ALG_H
