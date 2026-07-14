
// Determines whether to upgrade from a base HI-FI quant type to higher precision
// (e.g. Q6_K) by checking if the given index falls into critical regions of the
// total range: the bottom 1/8th, the top 1/8th, or every 3rd region in between.
bool use_more_bits(int i_total_weighted_value, int n_total) {
    return i_total_weighted_value < n_total / 8 ||
           i_total_weighted_value >= 7 * n_total / 8 ||
           (i_total_weighted_value - n_total / 8) % 3 == 2;
}
