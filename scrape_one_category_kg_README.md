This script is the Kyrgyz-optimized version of scrape_one_category_ru.py. While the core logic remains the same, it has been specifically tuned to handle the Kyrgyz language version of the website.

🔍 What is the difference?
There are three key technical differences in scrape_one_category_ru.py compared to the first one:

Language Headers: It sets accept_language to ky-KG, telling the website's server to serve the Kyrgyz version of the content.

Keyword Expansion: It includes Kyrgyz keywords like көбүрөөк (more), толугураак (fuller/more details), and башкы (home/main) to correctly identify interactive buttons and breadcrumbs.

Path Normalization: It includes logic to handle the /kg/ URL prefix specifically used for the Kyrgyz sections of the site.