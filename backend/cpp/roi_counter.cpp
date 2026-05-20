#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <omp.h>

// ─── Minimal JSON parser (no dependencies) ───────────────────────────────────
// Parses only the flat structure we produce from Python.
// Format expected:
// {
//   "rois": [ {"name":"Tramo 1","x1":0,"y1":0,"x2":100,"y2":100}, ... ],
//   "boxes": [ {"x1":10,"y1":10,"x2":30,"y2":30}, ... ],
//   "threads": 4
// }

struct Rect { float x1, y1, x2, y2; };
struct ROI  { std::string name; Rect rect; };

static std::string extractStr(const std::string& s, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = s.find(search);
    if (pos == std::string::npos) return "";
    pos = s.find('"', pos + search.size() + 1);
    if (pos == std::string::npos) return "";
    auto end = s.find('"', pos + 1);
    return s.substr(pos + 1, end - pos - 1);
}

static float extractFloat(const std::string& s, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = s.find(search);
    if (pos == std::string::npos) return 0.0f;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return 0.0f;
    return std::stof(s.substr(pos + 1));
}

static int extractInt(const std::string& full, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = full.find(search);
    if (pos == std::string::npos) return 1;
    pos = full.find(':', pos);
    if (pos == std::string::npos) return 1;
    return std::stoi(full.substr(pos + 1));
}

// Split a JSON array string "[{...},{...}]" into individual object strings
static std::vector<std::string> splitObjects(const std::string& arr) {
    std::vector<std::string> out;
    int depth = 0;
    std::string cur;
    for (char c : arr) {
        if (c == '{') { depth++; cur += c; }
        else if (c == '}') {
            cur += c;
            if (--depth == 0) { out.push_back(cur); cur.clear(); }
        } else if (depth > 0) { cur += c; }
    }
    return out;
}

// Extract a JSON array by key from a full JSON string
static std::string extractArray(const std::string& full, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = full.find(search);
    if (pos == std::string::npos) return "[]";
    pos = full.find('[', pos);
    if (pos == std::string::npos) return "[]";
    int depth = 0;
    std::string out;
    for (size_t i = pos; i < full.size(); ++i) {
        char c = full[i];
        out += c;
        if (c == '[') depth++;
        else if (c == ']') { if (--depth == 0) break; }
    }
    return out;
}

// ─── Geometry ────────────────────────────────────────────────────────────────

static bool boxInROI(const Rect& box, const Rect& roi) {
    // Center of bounding box must fall inside the ROI
    float cx = (box.x1 + box.x2) * 0.5f;
    float cy = (box.y1 + box.y2) * 0.5f;
    return cx >= roi.x1 && cx <= roi.x2 && cy >= roi.y1 && cy <= roi.y2;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {

    // Read full stdin into a string
    std::ostringstream buf;
    buf << std::cin.rdbuf();
    std::string input = buf.str();

    // Parse threads
    int nThreads = extractInt(input, "threads");
    omp_set_num_threads(nThreads);

    // Parse ROIs
    std::string roisArr = extractArray(input, "rois");
    auto roiObjs = splitObjects(roisArr);
    std::vector<ROI> rois;
    for (auto& obj : roiObjs) {
        ROI r;
        r.name       = extractStr(obj,   "name");
        r.rect.x1    = extractFloat(obj, "x1");
        r.rect.y1    = extractFloat(obj, "y1");
        r.rect.x2    = extractFloat(obj, "x2");
        r.rect.y2    = extractFloat(obj, "y2");
        rois.push_back(r);
    }

    // Parse boxes
    std::string boxesArr = extractArray(input, "boxes");
    auto boxObjs = splitObjects(boxesArr);
    std::vector<Rect> boxes;
    for (auto& obj : boxObjs) {
        Rect b;
        b.x1 = extractFloat(obj, "x1");
        b.y1 = extractFloat(obj, "y1");
        b.x2 = extractFloat(obj, "x2");
        b.y2 = extractFloat(obj, "y2");
        boxes.push_back(b);
    }

    int nROIs  = (int)rois.size();
    int nBoxes = (int)boxes.size();
    std::vector<int> counts(nROIs, 0);

    // ── Parallel count ────────────────────────────────────────────────────────
    // Each thread owns one or more ROIs and iterates over all boxes.
    // schedule(static) distributes ROIs evenly across threads.
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nROIs; ++i) {
        int c = 0;
        for (int j = 0; j < nBoxes; ++j) {
            if (boxInROI(boxes[j], rois[i].rect)) c++;
        }
        counts[i] = c;   // each i is unique → no race condition
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Output JSON ──────────────────────────────────────────────────────────
    std::cout << "{\n";
    std::cout << "  \"counts\": {\n";
    for (int i = 0; i < nROIs; ++i) {
        std::cout << "    \"" << rois[i].name << "\": " << counts[i];
        if (i < nROIs - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "  },\n";
    std::cout << "  \"total\": " << nBoxes << ",\n";
    std::cout << "  \"threads_used\": " << omp_get_max_threads() << ",\n";
    std::cout << "  \"elapsed_ms\": " << elapsed_ms << "\n";
    std::cout << "}\n";

    return 0;
}