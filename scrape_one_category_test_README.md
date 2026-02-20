# O-WebsiteScrapping

A robust, high-performance web crawler and scraper designed to extract structured content from complex websites.

## 🚀 Features
* **Dual-Layer Extraction:** Uses `httpx` for speed and `Playwright` for JS-heavy interactive elements.
* **Smart Discovery:** Automatically crawls sub-pages within a specific root subtree.
* **Interactive Handling:** Clicks "Read More" buttons, expands accordions, and handles lazy-loading via automated scrolling.
* **Noise Reduction:** Specifically tuned to strip out footers, legal disclaimers, and navigation menus to focus on main content.
* **Structured Output:** Saves data in `.jsonl` format for easy processing in Data Science pipelines.

## 🛠️ Installation

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/AiziliIsu/O-WebsiteScrapping.git](https://github.com/AiziliIsu/O-WebsiteScrapping.git)

```

2. **Install dependencies:**
```bash
pip install httpx trafilatura beautifulsoup4 tenacity playwright
playwright install chromium

```



## 📖 Usage

Run the script by providing the target category URL:

```bash
python scrape_one_category_test.py "[https://example.com/category-url](https://example.com/category-url)"

```

## 📁 Output

The script generates a `.jsonl` file in the configured output directory, containing:

* URL and Title
* Breadcrumb paths
* Full cleaned content text
* Short summary (About)

```

---

### Part 2: Terminal Commands (The "Initialization")

Follow these steps in your PyCharm terminal. I have included the command to create the README file directly from the terminal for you.



**1. Initialize and create the documentation:**
```bash
git init
# This creates the README file with the content I wrote above
echo "# O-WebsiteScrapping" > README.md

```

*(Note: You can also just right-click in PyCharm, create a New File named `README.md`, and paste my text from Part 1).*

**2. Link to your GitHub repository:**

```bash
git remote add origin https://github.com/AiziliIsu/O-WebsiteScrapping.git
git branch -M main

```

**3. Stage and Commit your files:**

```bash
git add README.md scrape_one_category_test.py
git commit -m "Initial commit: Added robust scraper with JS rendering and README"

```

**4. Push to GitHub:**

```bash
git push -u origin main

```

---

### Important Note on Security

In your code, you have a hardcoded path:
`OUTPUT_DIR = Path(r"C:\Users\1000001392\...")`

**Peer Advice:** Since you are pushing this to a public repository, other people won't have that folder on their computer. Would you like me to show you how to change that line so it creates a "results" folder automatically inside the project directory instead?