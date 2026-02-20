This script is the English-optimized version of scrape_one_category_ru.py. While the core logic remains the same, it has been specifically tuned to handle the English language version of the website.
* **Language-Specific Headers:** Forced `en-US` request headers to bypass regional auto-redirects.
* **CLI Flexibility:** Supports arguments for output paths and custom metadata tags.
* **Advanced Noise Filtering:** Strips headers and footers but preserves deep-nested content.

### Usage
Run the script via terminal:
```bash
python scrape_english_category.py "[https://o.kg/en/chastnym-klientam/](https://o.kg/en/chastnym-klientam/)" --lang en --out "./dataset/english"