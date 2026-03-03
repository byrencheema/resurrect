# Sample B&W Content for Testing Resurrect

Download these files and place them in this `samples/` directory for testing.

---

## Video Clips (Silent Films — Public Domain)

### Charlie Chaplin (pre-1927, all public domain)

| Film | Year | Link | Notes |
|------|------|------|-------|
| The Tramp | 1915 | https://archive.org/details/charlie-chaplin-.-the-tramp-1915-restored-silent-short-film-noir-comedy | Restored print. Great variety of scenes — outdoor, indoor, physical comedy. |
| One A.M. | 1916 | https://archive.org/details/CC_1916_08_07_One_A_M | Solo Chaplin performance. Excellent for motion-preservation colorization. |
| The Adventurer | 1917 | https://archive.org/details/CharlieChaplinTheAdventurer1917pianoScore | Fast-paced comedy. Beach + mansion scenes. |
| The Bank | 1915 | https://archive.org/details/1915bank | Indoor bank scenes, varied lighting. |
| Sunnyside | 1919 | https://archive.org/details/Sunnyside | Rural outdoor scenes. 28 min. |
| His Prehistoric Past | 1914 | https://archive.org/details/his-prehistoric-past-1914_202304 | Stone-age comedy, unique costumes. |
| Chaplin Collection | Various | https://archive.org/details/CharlieChaplin | 37 files, multiple formats. |
| Chaplin Festival (4 shorts) | 1917 | https://archive.org/details/charlie_chaplin_film_fest | The Adventurer, The Cure, Easy Street, The Immigrant. |
| Grand Archive | Various | https://archive.org/details/sircharliechaplin | Largest Chaplin collection on archive.org. |

### Other Silent Era Films

| Film | Year | Link | Notes |
|------|------|------|-------|
| Nosferatu | 1922 | https://archive.org/details/nosferatu_1922 | German Expressionist horror. High contrast B&W. |
| The Cabinet of Dr. Caligari | 1920 | https://archive.org/details/the-cabinet-of-dr-caligari | Surreal set design, dramatic lighting. |
| A Trip to the Moon | 1902 | https://archive.org/details/LeVoyageDansLaLune1902 | Georges Melies classic. Early cinema. |
| The General | 1926 | https://archive.org/details/TheGeneral1926 | Buster Keaton. Train chase sequences. Varied outdoor scenes. |
| Sherlock Jr. | 1924 | https://archive.org/details/SherlockJr1924 | Buster Keaton. Great for testing motion. |

### Historical Footage / Newsreels

| Content | Link | Notes |
|---------|------|-------|
| Prelinger Archives | https://archive.org/details/prelinger | 17,000+ items. Industrial, educational, advertising films. |
| Stock Footage Archive | https://archive.org/details/stock_footage | General historical footage. |
| Public Domain Archive | https://archive.org/details/public-domain-archive | Mixed public domain content. |

### How to Download from Archive.org

1. Go to the item page
2. On the right side, find "DOWNLOAD OPTIONS"
3. Choose "MPEG4" or "512Kb MPEG4" for MP4 format
4. Right-click → Save As, or click to download

For short clips: trim a longer film to 10-30 seconds with:
```bash
ffmpeg -i input.mp4 -ss 00:01:00 -t 30 -c copy short_clip.mp4
```

---

## Photographs (B&W — Public Domain)

### Library of Congress Collections

| Collection | Link | Notes |
|-----------|------|-------|
| Free to Use & Reuse | https://www.loc.gov/free-to-use/ | Curated public domain images. |
| Prints & Photographs Catalog | https://www.loc.gov/pictures/ | 15M+ items, 1M+ digitized. |
| FSA/OWI (1935-1944) | https://www.loc.gov/pictures/collection/fsa/ | 175,000 Depression-era photos. Dorothea Lange, Walker Evans, etc. |
| Civil War Photographs | https://www.loc.gov/pictures/collection/cwp/ | Thousands of Civil War-era photos. |
| Detroit Publishing Co. | https://www.loc.gov/pictures/collection/det/ | 1885-1930. Cityscapes, landscapes, architecture. |
| Edward S. Curtis Collection | https://www.loc.gov/pictures/collection/ecur/ | Native American photos, 1890-1929. |

### How to Download from LOC

1. Search or browse a collection
2. Click on an image
3. Look for "Download" links — choose JPEG or TIFF
4. Images with "LC-DIG" numbers are available digitally

### Other Photo Sources

| Source | Link | Notes |
|--------|------|-------|
| Unsplash (B&W tag) | https://unsplash.com/s/photos/black-and-white-vintage | Free high-res vintage B&W photos. |
| Wikimedia Commons | https://commons.wikimedia.org/wiki/Category:Black_and_white_photographs | Huge collection of public domain B&W. |
| National Archives | https://www.archives.gov/research/still-pictures | US government historical photos. |
| Shorpy (reference) | https://www.shorpy.com | High-res historical photos (check individual licensing). |

---

## Recommended Test Set

For a quick test of all Resurrect modes, download these:

1. **Video (short clip)**: Trim 15-30 seconds from "The Tramp" (1915) — good mix of indoor/outdoor, people, movement
2. **Video (action)**: Trim 10 seconds from "The General" (1926) — train chase, fast movement
3. **Photo (portrait)**: Any FSA/OWI portrait from LOC — faces, clothing, period detail
4. **Photo (cityscape)**: Detroit Publishing Co. city scene — buildings, streets, vehicles
5. **Photo (landscape)**: Ansel Adams landscape — dramatic B&W with strong tonal range

---

## Quick FFmpeg Commands

```bash
# Trim a video to 15 seconds starting at 1 minute
ffmpeg -i full_movie.mp4 -ss 00:01:00 -t 15 -c copy clip.mp4

# Convert to standard resolution for faster processing
ffmpeg -i clip.mp4 -vf "scale=640:-1" -c:a copy clip_small.mp4

# Extract a still frame for photo mode testing
ffmpeg -i clip.mp4 -ss 00:00:05 -vframes 1 frame.jpg

# Check if a video has audio (useful for testing merge logic)
ffprobe -v quiet -select_streams a -show_entries stream=codec_type clip.mp4
```
