#include <cassert>
# include <iostream>

#include "atlas/field/Field.h"
#include "atlas/field/FieldSet.h"
#include "atlas/array.h"
#include "atlas/array/MakeView.h"

using atlas::idx_t;

int main(int argc, char** argv) {
    const auto test_point = 3;
    const auto test_level = 2;

    const idx_t horizontal_count = 1024 * 1024;
    const idx_t vertical_count = 37;
    auto field = atlas::Field("x", atlas::make_datatype<double>(), atlas::array::make_shape(horizontal_count, vertical_count));

    auto h_view = atlas::array::make_host_view<double,2>(field);
    for(idx_t level = 0; level < h_view.shape<1>(); ++level) {
        for(idx_t i = 0; i < h_view.shape<0>(); ++i ) {
            h_view(i,level) = (i * h_view.shape<0>()) + (level * h_view.shape<1>());
        }
    }

    assert( h_view(test_point, test_level) == (test_point * h_view.shape<0>()) + (test_level * h_view.shape<1>()) );

    field.updateDevice();

    auto d_view = atlas::array::make_device_view<double,2>(field);
    const auto nlevels = d_view.shape<1>();
    const auto level_stride = d_view.stride<1>();
    const auto npoints = d_view.shape<0>();
    const auto point_stride = d_view.stride<0>();
    const auto d_ptr = d_view.data();
    {
        #pragma omp target is_device_ptr(d_ptr)
        #pragma omp teams loop
        for(idx_t i = 0; i < npoints; ++i ) {
            #pragma omp loop
            for(idx_t level = 0; level < nlevels; ++level) {
                d_ptr[(i * point_stride) + (level * level_stride)] += 10;
            }
        }
    }

    field.updateHost();

    assert( h_view(test_point, test_level) == (test_point * h_view.shape<0>()) + (test_level * h_view.shape<1>()) + 10 );

    return 0;
}
