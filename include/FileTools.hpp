

#ifndef FILETOOLS_HPP
#define FILETOOLS_HPP

#include <sys/stat.h>

bool createFolder(const char* folderPath) {
    struct stat info;
    if (stat(folderPath, &info) != 0) {
        // Folder does not exist, create it
        if (mkdir(folderPath, 0777) != 0) {
            // Failed to create folder
            std::cerr << "Error creating folder: " << folderPath << std::endl;
            return false;
        }
        std::cout << "Folder created: " << folderPath << std::endl;
    } else if (info.st_mode & S_IFDIR) {
        // Folder exists
        std::cout << "Folder already exists: " << folderPath << std::endl;
    } else {
        // Path exists but not a folder
        std::cerr << "Error: Path already exists but is not a folder: " << folderPath << std::endl;
        return false;
    }
    return true;
}


#endif