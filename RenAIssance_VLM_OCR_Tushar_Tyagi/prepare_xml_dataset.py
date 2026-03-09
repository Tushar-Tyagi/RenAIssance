import os
import glob
import json
import argparse
import xml.etree.ElementTree as ET
import re

def parse_page_xml(xml_path: str) -> tuple[str, str]:
    """
    Parses a PAGE XML file to extract the image filename and the full transcribed text.
    Extracts text by finding all Words within TextLines and joining them.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Dynamically extract namespace from the root element
    m = re.match(r'\{(.*)\}', root.tag)
    ns = {'ns': m.group(1)} if m else {}
    prefix = 'ns:' if ns else ''
    
    # Extract image filename
    page_node = root.find(f'.//{prefix}Page', ns)
    if page_node is not None and 'imageFilename' in page_node.attrib:
        image_filename = page_node.attrib['imageFilename']
    else:
        # Fallback to XML filename if imageFilename is missing
        image_filename = os.path.basename(xml_path).replace('.xml', '.jpg')
        
    extracted_lines = []
    
    # Iterate through each TextRegion -> TextLine
    for text_region in root.findall(f'.//{prefix}TextRegion', ns):
        for text_line in text_region.findall(f'.//{prefix}TextLine', ns):
            words = []
            # In PAGE XML, the text can be under Word/TextEquiv/Unicode
            for word in text_line.findall(f'.//{prefix}Word', ns):
                unicode_node = word.find(f'.//{prefix}Unicode', ns)
                if unicode_node is not None and unicode_node.text:
                    words.append(unicode_node.text.strip())
            
            if words:
                extracted_lines.append(" ".join(words))
            else:
                # Fallback: Check if TextLine has a direct TextEquiv/Unicode (no Word divisions)
                line_unicode = text_line.find(f'.//{prefix}Unicode', ns)
                if line_unicode is not None and line_unicode.text:
                    extracted_lines.append(line_unicode.text.strip())
                    
    return image_filename, "\n".join(extracted_lines)


def main():
    parser = argparse.ArgumentParser(description="Convert PAGE XML dataset to JSONL and TXT files for VLM fine-tuning.")
    parser.add_argument("--xml_dir", type=str, required=True, help="Directory containing the XML files.")
    parser.add_argument("--output_jsonl", type=str, default="data/train_annotations.jsonl", help="Output JSONL file path.")
    parser.add_argument("--save_txt", action="store_true", help="Also save individual .txt files alongside the original XMLs.")
    parser.add_argument("--image_dir_prefix", type=str, default="", help="Prefix to add to image filenames in JSONL.")
    
    args = parser.parse_args()
    
    xml_files = glob.glob(os.path.join(args.xml_dir, "**", "*.xml"), recursive=True)
    if not xml_files:
        print(f"No XML files found in {args.xml_dir}")
        return
        
    processed_count = 0
    
    # Ensure output directory exists before writing
    output_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        for xml_file in xml_files:
            try:
                image_filename, text_content = parse_page_xml(xml_file)
                
                # Combine prefix if given
                if args.image_dir_prefix:
                    image_filename = os.path.join(args.image_dir_prefix, image_filename)
                
                # 1. Write to JSONL
                record = {
                    "image": image_filename,
                    "text": text_content
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # 2. Optionally write to individual .txt file
                if args.save_txt:
                    txt_path = os.path.splitext(xml_file)[0] + ".txt"
                    with open(txt_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(text_content)
                
                processed_count += 1
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                
    print(f"Successfully processed {processed_count}/{len(xml_files)} XML files.")
    print(f"JSONL annotations saved to: {args.output_jsonl}")

if __name__ == "__main__":
    main()
