import os
from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html

def clean_html_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Retain only the main content area
    content_root = soup.find("div", id="mw-content-text")
    if content_root:
        soup.body.clear()
        soup.body.append(content_root)

    # Remove UI clutter
    for selector in [
        ".navbox", ".printfooter", ".toc", ".mw-editsection",
        ".mw-parser-output .hlist"
    ]:
        for tag in soup.select(selector):
            tag.decompose()

    # Remove junk strings
    for el in soup.find_all(string=lambda s: (
        "wiki." in s or
        "Retrieved from" in s or
        "In other languages:" in s
    )):
        el.extract()

    # Remove the entire "See also" section if it exists
    see_also_span = soup.find("span", id="See_also")
    if see_also_span:
        heading = see_also_span.find_parent("h2")
        if heading:
            current = heading
            while current:
                next_node = current.find_next_sibling()
                if next_node and next_node.name and next_node.name.startswith("h"):
                    break
                current.decompose()
                current = next_node
            heading.decompose()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(soup))

# Main script
input_dir = "raw-html"
temp_dir = "cleaned-html"
output_dir = "cleaned-markdown"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".html"):
        raw_path = os.path.join(input_dir, filename)
        cleaned_path = os.path.join(temp_dir, filename)
        clean_html_file(raw_path, cleaned_path)

        # Convert cleaned HTML to markdown
        elements = partition_html(filename=cleaned_path)
        content = "\n\n".join([el.text for el in elements])

        out_path = os.path.join(output_dir, filename.replace(".html", ".md"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

print("✅ HTML cleaned and converted to markdown.")
