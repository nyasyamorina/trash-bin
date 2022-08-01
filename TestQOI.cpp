#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <filesystem>
#include <stdint.h>
#include <iterator>
#include <stdio.h>

#define cimg_verbosity 1   // not output library messages on a basic dialog window
#define cimg_use_jpeg
#define cimg_use_png

#include "CImg.h"

namespace fs = std::filesystem;
namespace cim = cimg_library;

//constexpr uint8_t u8_2 = 2;


using Data = std::vector<uint8_t>;

using Img8 = cimg_library::CImg<uint8_t>;
void getcolor(Img8 const& im, uint32_t x, uint32_t y, uint8_t & r, uint8_t & g, uint8_t & b, uint8_t & a) {
    r = im(x, y, 0, 0);
    g = im(x, y, 0, 1);
    b = im(x, y, 0, 2);
    a = (im.spectrum() > 3) ? im(x, y, 0, 3) : 255;
}
void setcolor(Img8 & im, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    im(x, y, 0, 0) = r;
    im(x, y, 0, 1) = g;
    im(x, y, 0, 2) = b;
    if (im.spectrum() > 3) {
        im(x, y, 0, 3) = a;
    }
}
inline void getcolor(Img8 const& im, size_t idx, uint8_t & r, uint8_t & g, uint8_t & b, uint8_t & a) {
    uint32_t y = static_cast<uint32_t>(idx / im.width()), x = idx % im.width();
    getcolor(im, x, y, r, g, b, a);
}
inline void setcolor(Img8 & im, size_t idx, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    uint32_t y = static_cast<uint32_t>(idx / im.width()), x = idx % im.width();
    setcolor(im, x, y, r, g, b, a);
}

using ParentRelative = std::pair<fs::path, fs::path>;
std::ostream & operator << (std::ostream & out, ParentRelative const& pr) {
    char constexpr separator = fs::path::preferred_separator;      // convert separator from `wchar_t` to `char`
    return out << "\"<" << pr.first.string() << '>' << separator << pr.second.string() << '\"';
}



inline bool startswith(std::string const& a, std::string const& b) {
    return a.find(b) == 0;
}
inline bool endswith(std::string const& a, std::string const& b) {
    return a.rfind(b) == (a.length() - b.length());
}


void scan_files(std::vector<ParentRelative> & files, fs::path const& parent, fs::path const& relative) {
    fs::path curr = parent / relative;

    std::vector<fs::path> listdir;
    for (fs::directory_entry const& d_entry : fs::directory_iterator(curr)) {
        fs::path name = d_entry.path().filename();
        if (name == ".noimage") {
            return;
        }
        listdir.push_back(name);
    }

    for (fs::path const& name : listdir) {
        fs::path relname = relative / name;
        fs::path absname = curr / name;

        if (fs::is_directory(absname)) {
            scan_files(files, parent, relname);
        }
        else if (fs::is_regular_file(absname)) {
            files.push_back(ParentRelative(parent, relname));
        }
        else {
            std::cout << "\033[33munknown path " << absname << "\033[0m" << std::endl;
        }
    }
}


fs::path convert_ext(fs::path const& file, std::string const& ext) {
    std::string filepath = file.string();
    size_t idx = filepath.rfind('.');
    idx = (idx == std::string::npos) ? filepath.length() : idx;
    return filepath.substr(0, idx) + ext;
}

void print_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    ::std::cout << '[' << r << ' ' << g << ' ' << b << ']';
}


namespace nyas {

    class Processbar {
    public:
        size_t n;
        size_t curr;
        size_t next_ratio1000;

        Processbar(size_t const& n_) :
            n(n_), curr(0), next_ratio1000(0) {
            this->update(0);
        }

        void update(size_t add = 1) {
            this->curr += add;
            size_t ratio1000 = 1000 * this->curr / this->n;
            if (ratio1000 > 1000) {
                return;
            }
            if (ratio1000 >= this->next_ratio1000) {
                ++this->next_ratio1000;
                size_t done = (ratio1000 + 5) / 10;
                size_t prec = ratio1000 / 10, rem = ratio1000 % 10;

                ::std::cout << '[';
                for (int _ = 1; _ < done; ++_) {
                    ::std::cout << '=';
                }
                if (done > 0) {
                    ::std::cout << '>';
                }
                for (int _ = 100; _ > done; --_) {
                    ::std::cout << ' ';
                }
                ::std::cout << "] " << prec << '.' << rem << '%' << '\r' << ::std::flush;
            }
        }
    };

    void print_help() {
        ::std::cout << "A simple QOI (the Quite OK Image format) en-decoder is implemented by nyasyamorina" << ::std::endl;
        ::std::cout << ::std::endl;
        ::std::cout << "it will auto convert all images from jpeg/png to qoi, or from qoi to png." << ::std:: endl;
        ::std::cout << "you can put file named \".noimage\" in directory to specify not convert images in this directory." << ::std::endl;
        ::std::cout << ::std::endl;
        ::std::cout << "\033[32musage: TestQOI [infile / indir]...\033[0m" << ::std::endl << ::std::endl;
    }


    bool is_qoi(fs::path const& filepath) {
        ::std::ifstream file(filepath, ::std::ios::binary);
        if (!file) {
            return false;//endswith(filepath.string(), ".qoi");
        }
        char qoif[5] = {0, 0, 0, 0, 0};
        file.read(qoif, 4);
        file.close();
        return ::std::string(qoif) == "qoif";
    }


    size_t index_position(size_t r, size_t g, size_t b, size_t a) {
        return (3 * r + 5 * g + 7 * b + 11 * a) % 64;
    }

    Data qoi_encode(Img8 const& im, uint32_t & out_channels) {
        Data data;

        uint32_t width = im.width(), height = im.height();
        out_channels = im.spectrum();
        uint8_t channels = static_cast<uint8_t>(out_channels);
        if (out_channels != 3 && out_channels != 4) {
            return data;
        }
        data.reserve(static_cast<size_t>(width) * height * channels);

        // write header
        data.push_back('q');
        data.push_back('o');
        data.push_back('i');
        data.push_back('f');
        data.push_back(width >> 24);
        data.push_back((width >> 16) & 0xFF);
        data.push_back((width >> 8) & 0xFF);
        data.push_back(width & 0xFF);
        data.push_back(height >> 24);
        data.push_back((height >> 16) & 0xFF);
        data.push_back((height >> 8) & 0xFF);
        data.push_back(height & 0xFF);
        data.push_back(channels);
        data.push_back(1);      // colorspace

        uint8_t running_r[64] = {0};
        uint8_t running_g[64] = {0};
        uint8_t running_b[64] = {0};
        uint8_t running_a[64] = {0};
        uint8_t prev_r, prev_g, prev_b, prev_a;
        uint8_t curr_r = 0, curr_g = 0, curr_b = 0, curr_a = 255;
        size_t running_idx = 0;

        size_t pixel_idx = 0, total_pixels = static_cast<size_t>(width) * height;

        while (pixel_idx < total_pixels) {
            prev_r = running_r[running_idx] = curr_r;
            prev_g = running_g[running_idx] = curr_g;
            prev_b = running_b[running_idx] = curr_b;
            prev_a = running_a[running_idx] = curr_a;

            getcolor(im, pixel_idx++, curr_r, curr_g, curr_b, curr_a);
            running_idx = index_position(curr_r, curr_g, curr_b, curr_a);

            if (curr_r == prev_r && curr_g == prev_g &&
                curr_b == prev_b && curr_a == prev_a) {                 // -> QOI_OP_RUN

                --pixel_idx;
                uint8_t run = 0;
                bool same_next = true;
                while (same_next && run < 0x3E) {
                    ++run;
                    same_next = (++pixel_idx) < total_pixels;
                    if (same_next) {
                        getcolor(im, pixel_idx, curr_r, curr_g, curr_b, curr_a);
                        same_next = curr_r == prev_r && curr_g == prev_g &&
                                    curr_b == prev_b && curr_a == prev_a;
                    }
                }
                curr_r = prev_r; curr_g = prev_g;
                curr_b = prev_b; curr_a = prev_a;
                data.push_back(0xC0 | (run - 1));
                continue;
            }
            else if (curr_r == running_r[running_idx] && curr_g == running_g[running_idx] &&
                     curr_b == running_b[running_idx] && curr_a == running_a[running_idx]) {        // -> QOI_OP_INDEX
                data.push_back(static_cast<uint8_t>(running_idx));
                continue;
            }
            else if (curr_a == prev_a) {
                uint8_t dr = curr_r - prev_r + 2, dg = curr_g - prev_g + 2, db = curr_b - prev_b + 2;
                if (dr < 4 && dg < 4 && db < 4) {                       // -> QOI_OP_DIFF
                    data.push_back(0x40 | (dr << 4) | (dg << 2) | db);
                    continue;
                }
                dr += 8 - dg;
                db += 8 - dg;
                dg += 30;
                if (dr < 16 && dg < 64 && db < 16) {                    // -> QOI_OP_LUMA
                    data.push_back(0x80 | dg);
                    data.push_back((dr << 4) | db);
                    continue;
                }
            }

            data.push_back((channels == 3) ? 0xFE : 0xFF);
            data.push_back(curr_r);
            data.push_back(curr_g);
            data.push_back(curr_b);
            if (channels == 4) {
                data.push_back(curr_a);
            }
        }

        // write end of file
        data.push_back(0);
        data.push_back(0);
        data.push_back(0);
        data.push_back(0);
        data.push_back(0);
        data.push_back(0);
        data.push_back(0);
        data.push_back(1);

        data.shrink_to_fit();
        return data;
    }

    Img8 qoi_decode(Data const& data) {
        // read header
        uint32_t width = 0, height = 0;
        width |= static_cast<uint32_t>(data[4]) << 24;
        width |= static_cast<uint32_t>(data[5]) << 16;
        width |= static_cast<uint32_t>(data[6]) << 8;
        width |= static_cast<uint32_t>(data[7]);
        height |= static_cast<uint32_t>(data[8]) << 24;
        height |= static_cast<uint32_t>(data[9]) << 16;
        height |= static_cast<uint32_t>(data[10]) << 8;
        height |= static_cast<uint32_t>(data[11]);
        uint8_t channels = data[12];
        if (channels != 3 && channels != 4) {
            ::std::cout << "\033[31mqoi decode channels fails at 12-th byte\033[0m" << ::std::endl;
            return Img8::empty();
        }

        // init image
        Img8 im(width, height, 1, channels, 0);

        uint8_t running_r[64]{};
        uint8_t running_g[64]{};
        uint8_t running_b[64]{};
        uint8_t running_a[64]{};
        uint8_t curr_r = 0, curr_g = 0, curr_b = 0, curr_a = 255;

        size_t byte_idx = 14, total_bytes = data.size();
        size_t pixel_idx = 0, total_pixels = static_cast<size_t>(width) * height;

        while (byte_idx < total_bytes && pixel_idx < total_pixels) {
            uint8_t byte = data[byte_idx++];
            uint8_t tag = byte & 0xC0, rem = byte & 0x3F;

            if (tag == 0xC0) {
                if (byte == 0xFE) {             // -> QOI_OP_RGB
                    if (byte_idx + 2 >= total_bytes) {
                        ::std::cout << "\033[33mqoi decode QOI_OP_RGB fails at " << byte_idx << "-th byte\033[0m" << ::std::endl;
                        break;
                    }
                    curr_r = data[byte_idx++];
                    curr_g = data[byte_idx++];
                    curr_b = data[byte_idx++];
                }
                else if (byte == 0xFF) {        // -> QOI_OP_RGBA
                    if (byte_idx + 3 >= total_bytes) {
                        ::std::cout << "\033[33mqoi decode QOI_OP_RGBA fails at " << byte_idx << "-th byte\033[0m" << ::std::endl;
                        break;
                    }
                    curr_r = data[byte_idx++];
                    curr_g = data[byte_idx++];
                    curr_b = data[byte_idx++];
                    curr_a = data[byte_idx++];
                }
                else {                          // -> QOI_OP_RUN
                    while (pixel_idx + 1 < total_pixels && (rem--) > 0) {
                        setcolor(im, pixel_idx++, curr_r, curr_g, curr_b, curr_a);
                    }
                }
            }
            else if (tag == 0) {                // -> QOI_OP_INDEX
                curr_r = running_r[rem];
                curr_g = running_g[rem];
                curr_b = running_b[rem];
                curr_a = running_a[rem];
            }
            else {
                uint8_t dr, dg, db;
                if (tag == 0x40) {              // -> QOI_OP_DIFF
                    dr = (rem >> 4) - 2;
                    dg = ((rem >> 2) & 0x3) - 2;
                    db = (rem & 0x3) - 2;
                }
                else {                          // -> QOI_OP_LUMA
                    if (byte_idx >= total_bytes) {
                        ::std::cout << "\033[33mqoi decode QOI_OP_LUMA fails at " << byte_idx << "-th byte \033[0m" << ::std::endl;
                        break;
                    }
                    byte = data[byte_idx++];
                    dg = rem - 32;
                    dr = (byte >> 4) - 8 + dg;
                    db = (byte & 0xF) - 8 + dg;
                }
                curr_r += dr;
                curr_g += dg;
                curr_b += db;
            }

            setcolor(im, pixel_idx++, curr_r, curr_g, curr_b, curr_a);
            size_t running_idx = index_position(curr_r, curr_g, curr_b, curr_a);
            running_r[running_idx] = curr_r;
            running_g[running_idx] = curr_g;
            running_b[running_idx] = curr_b;
            running_a[running_idx] = curr_a;
        }

        return im;
    }


    Img8 qoi_load(fs::path const& filepath) {
        ::std::ifstream file(filepath, ::std::ios::binary);
        if (!file) {
            ::std::cout << "\033[31mcannot open file " << filepath << "\033[0m" << ::std::endl;
            return Img8::empty();
        }
        Data data(::std::istreambuf_iterator<char>(file), {});
        file.close();
        if (data.size() < 15) {
            ::std::cout << "\033[31minvalid qoi file " << filepath << "\033[0m" << ::std::endl;
            return Img8::empty();
        }
        return qoi_decode(data);
    }

    void qoi_save(fs::path const& filepath, Img8 const& im) {
        uint32_t channels;
        Data data = qoi_encode(im, channels);
        if (data.size() == 0) {
            ::std::cout << "\033[31mqoi encode fails: get " << im.spectrum() << " channels image: " << filepath << "\033[0m" << ::std::endl;
            return;
        }
        ::std::ofstream file(filepath, ::std::ios::binary);
        if (!file) {
            ::std::cout << "\033[31mcannot open file " << filepath << "\033[0m" << ::std::endl;
            return;
        }
        file.write(reinterpret_cast<char const*>(&data[0]), data.size());
        file.close();
    }
}



int main(int argc, char const** argv) {
    if (argc == 1) {
        nyas::print_help();
        return 0;
    }

    std::vector<ParentRelative> infiles;
    for (int i = 1; i < argc; ++i) {
        fs::path inpath = argv[i];

        if (fs::is_directory(inpath)) {
            scan_files(infiles, inpath.parent_path(), inpath.filename());
        }
        else if (fs::is_regular_file(inpath)) {
            infiles.push_back(ParentRelative(inpath.parent_path(), inpath.filename()));
        }
        else {
            std::cout << "\033[33munknown path " << inpath << "\033[0m" << std::endl;
        }
    }

    fs::path curr = fs::current_path() / "TestQOI-outputs";

    nyas::Processbar prog(infiles.size());
    for (ParentRelative const& infile : infiles) {
        fs::path infilepath = infile.first / infile.second;

        if (nyas::is_qoi(infilepath)) {
            fs::path outfilepath = curr / convert_ext(infile.second, ".png");

            Img8 im = nyas::qoi_load(infilepath);
            if (!im.is_empty()) {
                fs::create_directories(outfilepath.parent_path());

                FILE * outfile;
                fopen_s(&outfile, outfilepath.string().c_str(), "wb");
                if (outfile == nullptr) {
                    std::cout << "\033[31mcannot open file " << outfilepath << "\033[0m" << std::endl;
                }
                else {
                    im.save_png(outfile);
                    fclose(outfile);
                }
            }
        }

        else {
            fs::path outfilepath = curr / convert_ext(infile.second, ".qoi");
            fs::create_directories(outfilepath.parent_path());

            try {
                Img8 im(infilepath.string().c_str());
                nyas::qoi_save(outfilepath, im);
            }
            catch (cim::CImgIOException const& error) {
                //std::cout << "\033[31mCImgIOException: \n" << error.what() << "\033[0m" << std::endl;
            }
        }
        prog.update();
    }

    std::cout << std::endl;
    system("pause");
}
