# src/svg_color_tool/core/svg_processor.py

import re
import time
import numpy as np
import xml.etree.ElementTree as ET
import math
import json 
import logging
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any, Union, Iterator

from svg_color.config import *
from svg_color.domain.models import ColorConfig

from svg_color.utils import geo_utils, svg_utils, color_utils
from svg_color.core.svg_element_parser import SvgElementParser
import cv2



logger = logging.getLogger(__name__)



class SvgProcessor:
    """
    Core logic for processing SVG content by applying colors sampled from image data.

    This class takes SVG content (string/bytes) and image data (NumPy array)
    and returns a modified ElementTree object representing the colored SVG.
    It is independent of file I/O and user interface frameworks.

    Attributes:
        max_colors (int): The maximum number of distinct colors in the final palette.
        color_threshold (float): The threshold for merging similar colors.
    """
    def __init__(self, color_config = None):
        """
        Initializes the SvgProcessor with configuration settings.

        Args:
            max_colors: Maximum number of distinct colors in the output palette.
                        Must be a positive integer. Defaults to 80.
            color_threshold: Euclidean distance threshold for merging similar colors.
                             Lower values merge less, higher values merge more.
                             Must be a non-negative number. Defaults to 30.

        Raises:
            ValueError: If max_colors or color_threshold have invalid values.
        """
        
        if not isinstance(color_config.max_colors, int) or color_config.max_colors <= 0:
            raise ValueError("max_colors must be a positive integer.")
        if not isinstance(color_config.color_threshold, (int, float)) or color_config.color_threshold < 0:
             raise ValueError("color_threshold must be a non-negative number.")

        # self.color_config = color_config
        self.set_color_config(color_config)
        #self.max_colors = max_colors
        # self.color_threshold = color_threshold
        self.svg_element_parser = SvgElementParser()

        # Register SVG namespaces globally for ElementTree writing convenience
        # This avoids 'ns0:' prefixes in the output SVG.
        ET.register_namespace('', _SVG_NS.strip('{}'))
        ET.register_namespace('xlink', _XLINK_NS.strip('{}'))

        logger.info(f"SvgProcessor initialized: max_colors={self.max_colors}, color_threshold={self.color_threshold}")

    def set_color_config(self, color_config):
        self.max_colors = color_config.max_colors
        self.color_threshold = color_config.color_threshold

    def _find_element_representative_color(self, img_data: np.ndarray, elem: ET.Element) -> Optional[Tuple[int, int, int]]:
        """
        High-level function to find the representative color for a single SVG element.

        Orchestrates coordinate extraction, representative point calculation,
        and color sampling for the given element.

        Args:
            img_data: The image data (RGB NumPy array).
            elem: The SVG element (e.g., path, polygon).

        Returns:
            The representative (R, G, B) color tuple, or None if any step fails.
        """
        # tag = ET.QName(elem.tag).localname
        tag = elem.tag.split('}')[-1]
        logger.debug(f"Finding color for element: <{tag}>")

        # 1. Get Coordinates
        coords = self.svg_element_parser.parse(elem)
        if not coords:
            logger.debug(f"Could not get coordinates for <{tag}>. Skipping color extraction.")
            return None
        
        

        # # 2. Calculate Representative Point
        # rep_point = geo_utils.bbox_calculate_representative_point(coords)
        # if not rep_point:
        #     logger.debug(f"Could not calculate representative point for <{tag}>. Skipping color extraction.")
        #     return None

        # # 3. Sample Color at Point
        # color = geo_utils.sample_color_at_point(img_data, rep_point)
        # if not color:
        #     logger.warning(f"Could not sample color at point {rep_point} for <{tag}>. Skipping.")
        #     return None

        # color = geo_utils.precise_sample_color_at_point(img_data, rep_point)


        color = color_utils.get_dominant_color(img_data, coords)
        
        logger.debug(f"Found representative color {color} for <{tag}>.")
        return color

    # --- Color Palette Generation ---

    def _extract_colors(self, root: ET.Element, img_data: np.ndarray) -> Dict[ET.Element, Tuple[int, int, int]]:
        """
        Extracts representative colors for all processable elements in the SVG tree.

        Args:
            root: The root Element of the parsed SVG tree.
            img_data: The image data (RGB NumPy array).

        Returns:
            A dictionary mapping each successfully processed Element to its
            extracted (R, G, B) color tuple.
        """
        # Build XPath query to find all relevant elements
        # elements_xpath = " | ".join([f".//{_SVG_NS}{tag}" for tag in _ELEMENTS_TO_PROCESS])
        # print(elements_xpath)
        # svg_ns = "{http://www.w3.org/2000/svg}"
        # print(f".//{svg_ns}path", f".//{svg_ns}polygon")
        # elements_to_process = root.findall(f".//{svg_ns}path") + root.findall(f".//{svg_ns}polygon")
        # print(f"INFO: Tìm thấy {len(elements_to_process)} phần tử <path>/<polygon>.")


        
        try:
            elements_to_process = []
            for tag in _ELEMENTS_TO_PROCESS:
                found_elements = root.findall(f".//{_SVG_NS}{tag}")
                # print(found_elements)
                elements_to_process.extend(found_elements)
            # elements_to_process = root.findall(elements_xpath)
   
        except SyntaxError as e:
             logger.error(f"Error in finding elements of SVG. Error: {e}")
             return {} # Return empty if query fails

        logger.info(f"Found {len(elements_to_process)} potentially colorable elements "
                    f"({', '.join(_ELEMENTS_TO_PROCESS)}). Starting color extraction...")

        element_color_map: Dict[ET.Element, Tuple[int, int, int]] = {}
        processed_count = 0
        skipped_count = 0
        start_extract_time = time.time()

        # Iterate and extract color for each element
        for i, elem in enumerate(elements_to_process):
            color = self._find_element_representative_color(img_data, elem)
            if color:
                element_color_map[elem] = color
                processed_count += 1
            else:
                 skipped_count += 1

            # Log progress periodically to avoid flooding logs
            if (i + 1) % 500 == 0 or i == len(elements_to_process) - 1:
                 elapsed = time.time() - start_extract_time
                 rate = (i + 1) / elapsed if elapsed > 0 else 0
                 logger.info(
                     f"Color extraction progress: {i+1}/{len(elements_to_process)} "
                     f"({rate:.1f} elem/s, {processed_count} colored, {skipped_count} skipped)"
                 )

        end_extract_time = time.time()
        logger.info(
            f"Color extraction completed in {end_extract_time - start_extract_time:.2f}s. "
            f"Extracted colors for {processed_count} elements."
        )
        return element_color_map

    def _merge_colors_simple(self,
                             all_colors: List[Tuple[int, int, int]]
                             ) -> List[Tuple[int, int, int]]:
        """
        Merges similar colors from a list into a final palette.

        Uses a simple greedy approach: sorts unique colors by frequency,
        then iteratively adds colors to the palette if they are sufficiently
        different from already added colors, up to `self.max_colors`.

        Args:
            all_colors: A list containing all extracted (R, G, B) color tuples.

        Returns:
            A sorted list representing the final color palette, containing at most
            `self.max_colors` distinct colors.
        """
        if not all_colors:
            logger.warning("No colors provided for merging, returning empty palette.")
            return []

        # Count occurrences of each unique color
        color_counts = Counter(all_colors)
        # Sort unique colors: primarily by frequency (most frequent first),
        # secondarily by the color tuple itself (for stable sorting)
        unique_sorted_by_freq = sorted(color_counts.keys(), key=lambda c: (-color_counts[c], c))
        num_unique = len(unique_sorted_by_freq)

        logger.info(
            f"Starting simple color merge for {num_unique} unique colors "
            f"(Threshold={self.color_threshold}, Max={self.max_colors})."
        )

        # If already within limit, no merging needed, just return sorted list
        if num_unique <= self.max_colors:
            logger.info("Number of unique colors is within limit. No merging necessary.")
            # Sort by color value for consistent output palette order
            return sorted(unique_sorted_by_freq)

        # --- Greedy Merging Process ---
        merged_palette: List[Tuple[int, int, int]] = []
        for color in unique_sorted_by_freq:
            # Stop if the palette limit is reached
            if len(merged_palette) >= self.max_colors:
                logger.debug(f"Reached max_colors ({self.max_colors}). Stopping merge process.")
                break

            # Check if this color is 'too close' to any color already in the palette
            is_too_close = False
            for existing_color in merged_palette:
                dist = color_utils.color_distance(color, existing_color) # Assumes color_utils.color_distance is imported
                if dist <= self.color_threshold:
                    is_too_close = True
                    # logger.debug(f"Color {color} is close to existing {existing_color} (dist={dist:.2f}). Skipping.")
                    break # No need to check further distances for this color

            # If the color is distinct enough, add it to the final palette
            if not is_too_close:
                # logger.debug(f"Adding distinct color {color} to palette. Size: {len(merged_palette) + 1}")
                merged_palette.append(color)

        # Final sort by color value for a consistent output order
        final_palette = sorted(merged_palette)
        logger.info(f"Simple merge resulted in {len(final_palette)} final colors in palette.")
        return final_palette

    def _merge_colors_improved(self, colors):
        """
        Merge similar colors based on color distance with improved logic

        Args:
            colors (list): List of RGB color tuples

        Returns:
            list: Merged colors list
        """
        # Nếu số lượng màu nhỏ hơn hoặc bằng max_colors, trả về nguyên danh sách
        if len(colors) <= self.max_colors:
            return colors

        # Sắp xếp các màu theo độ phổ biến (số lần xuất hiện)
        color_counts = Counter(colors)
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

        # Bắt đầu với các màu phổ biến nhất
        merged_colors = [color for color, _ in sorted_colors[:self.max_colors]]
        # print(len(merged_colors))
        # Nếu vẫn chưa đủ max_colors, thử thêm các màu khác
        if len(merged_colors) < self.max_colors:
            remaining_colors = [color for color, _ in sorted_colors[self.max_colors:]]

            for color in remaining_colors:
                # Kiểm tra xem màu mới có gần với bất kỳ màu nào trong merged_colors không
                is_similar = any(color_utils.color_distance(color, existing_color) <= self.color_threshold
                                for existing_color in merged_colors)

                if not is_similar and len(merged_colors) < self.max_colors:
                    merged_colors.append(color)
        
        return merged_colors
    
    # --- SVG Styling ---

    def _apply_styles(self,
                      root: ET.Element,
                      element_color_map: Dict[ET.Element, Tuple[int, int, int]],
                      final_palette: List[Tuple[int, int, int]]
                      ) -> int:
        """
        Applies the final color palette to SVG elements using CSS classes and attributes.

        Iterates through elements with extracted colors, finds the closest color
        in the final palette, and updates the element's 'fill', 'style', and 'class'
        attributes accordingly. Also updates/creates the main CSS <style> tag.

        Args:
            root: The root Element of the SVG tree to modify.
            element_color_map: Dictionary mapping elements to their original extracted colors.
            final_palette: The finalized list of (R, G, B) colors after merging.

        Returns:
            The number of elements whose styles were successfully updated.
            Returns 0 if the final palette is empty.
        """
        if not final_palette:
            logger.warning("Cannot apply styles: Final color palette is empty.")
            return 0

        # --- Prepare Mappings ---
        # Map each final color to a unique, stable CSS class name (st0, st1, ...)
        color_to_class = {color: f"st{idx}" for idx, color in enumerate(final_palette)}
        # Map the class name to its corresponding HEX color value
        class_to_hex = {cls: color_utils.rgb_to_hex(color) for color, cls in color_to_class.items()} # Assumes color_utils.rgb_to_hex imported

        logger.info(f"Applying styles using {len(final_palette)} final colors and classes (st0 to st{len(final_palette)-1}).")
        processed_count = 0

        # --- Iterate and Apply Styles to Elements ---
        for elem, original_color in element_color_map.items():
            # Skip if original color somehow wasn't found (shouldn't happen)
            if original_color is None:
                logger.warning(f"Skipping element {ET.QName(elem.tag).localname}: missing original color in map.")
                continue

            # Find the color in the final palette closest to the element's original color
            # The key function calculates distance from original_color to each palette color
            closest_final_color = min(final_palette,
                                      key=lambda pal_color: color_utils.color_distance(original_color, pal_color))

            # Get the corresponding hex value and CSS class name
            hex_color = color_utils.rgb_to_hex(closest_final_color)
            class_name = color_to_class[closest_final_color]

            # --- Modify Element Attributes (In-Place) ---
            # 1. Set 'fill' attribute directly: Ensures basic rendering even if CSS is ignored.
            elem.set("fill", hex_color)

            # 2. Clean 'style' attribute: Remove any pre-existing 'fill' property
            #    to avoid conflicts with the direct fill or CSS class.
            style_attr = elem.get("style")
            if style_attr:
                # Split into individual style declarations, strip whitespace
                style_parts = [p.strip() for p in style_attr.split(';') if p.strip()]
                # Keep only declarations that *don't* start with 'fill:' (case-insensitive)
                new_style_parts = [p for p in style_parts if not p.lower().startswith("fill:")]
                # Reconstruct the style attribute if other styles remain
                if new_style_parts:
                    elem.set("style", "; ".join(new_style_parts) + ";") # Use standard formatting
                # If 'fill' was the only style, remove the attribute entirely
                elif "style" in elem.attrib:
                    del elem.attrib["style"]

            # 3. Update 'class' attribute: Remove old 'st<number>' classes and add the new one.
            current_classes = elem.get("class", "").split()
            # Filter out empty strings and previous 'st' classes (using regex for safety)
            filtered_classes = [c for c in current_classes if c and not re.fullmatch(r'st\d+', c)]
            # Add the new class if it's not already somehow present
            if class_name not in filtered_classes:
                filtered_classes.append(class_name)
            # Set the updated class list, joined by spaces
            elem.set("class", " ".join(filtered_classes))

            processed_count += 1

        logger.info(f"Applied styles to {processed_count} individual SVG elements.")

        # --- Update/Create <style> tag in the SVG root ---
        svg_utils.update_or_create_style_tag(root, class_to_hex)

        return processed_count

    # --- Public Processing Method ---

    def process(self, svg_content: Union[str, bytes], image_data: np.ndarray) -> Optional[ET.ElementTree]:
        """
        Processes SVG content using image data to apply representative colors.

        This is the main public method orchestrating the entire workflow:
        1. Prepares the input image (ensures RGB format).
        2. Parses the input SVG content into an ElementTree.
        3. Extracts representative colors for each relevant SVG element by sampling the image.
        4. Merges the extracted colors into a limited, final color palette.
        5. Applies the final palette to the SVG elements using CSS classes and attributes.
        6. Returns the modified ElementTree object.

        Args:
            svg_content: The SVG content, either as a UTF-8 encoded string or bytes.
            image_data: The image data loaded as a NumPy array (expects BGR or BGRA format,
                        typically from `cv2.imread(..., cv2.IMREAD_UNCHANGED)`).

        Returns:
            An `xml.etree.ElementTree.ElementTree` object representing the processed
            SVG structure if successful. Returns the *original* parsed tree if no
            colors could be extracted (Step 3 fails partially). Returns `None` if
            a critical error occurs during image preparation, SVG parsing, or
            color merging (Steps 1, 2, or 4 fail).
        """

        logger.info("--- St arting SVG Processing Pipeline ---")
        start_process_time = time.time()

        # --- Step 1: Prepare Image Data ---
        logger.info("Step 1/5: Preparing image data...")
        img_rgb = image_data

        
        # --- Step 2: Parse SVG Content ---
        logger.info("Step 2/5: Parsing SVG content...")
        tree: Optional[ET.ElementTree] = None
        root: Optional[ET.Element] = None
        
        root, tree = svg_utils.parse_svg_content(svg_content)
        


        # --- Step 3: Extract Colors from Elements ---
        logger.info("Step 3/5: Extracting colors from SVG elements...")
        element_color_map = self._extract_colors(root, img_rgb)

        # If no colors could be extracted AT ALL, it might indicate a fundamental issue
        # (e.g., SVG shapes don't overlap image, image is blank, shapes have no area).
        # We return the original parsed tree in this case, as no modifications can be made.
        if not element_color_map:
            logger.warning("No representative colors were extracted from any SVG elements. "
                           "Check SVG/image alignment and content. Returning original SVG structure.")
            return tree # Return unmodified tree


        # --- Step 4: Merge Similar Colors ---
        logger.info("Step 4/5: Merging extracted colors into final palette...")
        all_extracted_colors = list(element_color_map.values())
        final_palette = self._merge_colors_improved(all_extracted_colors)

        # If merging somehow results in an empty palette (e.g., threshold too high, only one color extracted?)
        # this is likely an error state.
        if not final_palette:
            logger.error("Color merging process resulted in an empty final palette. Aborting.")
            return None # Critical failure


        # --- Step 5: Apply Final Styles ---
        logger.info("Step 5/5: Applying final styles to SVG tree...")
        self._apply_styles(root, element_color_map, final_palette)


        # --- Step 6: Return Processed Tree ---
        end_process_time = time.time()
        total_duration = end_process_time - start_process_time
        logger.info(f"--- SVG Processing Pipeline Finished Successfully in {total_duration:.2f}s ---")
        # The 'tree' object now contains the modified SVG structure
        return tree


if __name__ == "__main__":
    # --- 1. Cấu hình Logging cơ bản để xem output ---
    # (Trong ứng dụng thực tế, bạn sẽ cấu hình logging phức tạp hơn)
    logging.basicConfig(
        level=logging.INFO, # Đổi thành DEBUG để xem log chi tiết hơn từ processor
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    logger.info("--- Running SvgProcessor Example ---")

    # --- 2. Chuẩn bị Dữ liệu Input Mẫu ---

    from PIL import Image
    id = "008015"
    
    # Đọc ảnh PNG
    img = Image.open(f'/home/ansible/AI-Workspaces/Loc/Artasy/3_SVG-Coloring_beta/data/456789/input/037009.png').convert("RGB")
    sample_image_data = np.array(img)
    with open(f'/home/ansible/AI-Workspaces/Loc/Artasy/3_SVG-Coloring_beta/data/456789/input/037009.svg', 'r', encoding='utf-8') as f:
        sample_svg_string = f.read()


    # --- 3. Khởi tạo SvgProcessor ---
    # Sử dụng các giá trị cấu hình khác nhau để thử nghiệm

    config = ColorConfig(80, 30)
    processor = SvgProcessor(config)

    

    # --- 4. Gọi phương thức process ---
    logger.info("Calling processor.process()...")
    start_time = time.time()
    processed_tree: Optional[ET.ElementTree] = None # Khởi tạo biến

    # Đảm bảo cv2 đã được import thành công
    if cv2 is None:
        logger.critical("OpenCV (cv2) is not installed. Cannot run the example.")
    else:
        try:
            processed_tree = processor.process(sample_svg_string, sample_image_data)
        except Exception as e:
            logger.exception(f"An unexpected error occurred during processing: {e}")

    end_time = time.time()
    logger.info(f"Processing finished in {end_time - start_time:.2f} seconds.")

    # --- 5. Xử lý Kết quả ---
    if processed_tree:
        logger.info("Processing successful! Processed SVG Tree generated.")

        # Lấy phần tử gốc để làm việc
        processed_root = processed_tree.getroot()

        # # Cách 1: In ra nội dung XML của SVG đã xử lý (dạng string)
        # try:
        #     # encoding='unicode' để lấy string thay vì bytes
        #     # method='xml' để đảm bảo đúng chuẩn XML
        #     processed_svg_string = ET.tostring(processed_root, encoding='unicode', method='xml')
        #     print("\n--- Processed SVG Output ---")
        #     print(processed_svg_string)
        #     print("--------------------------\n")
        # except Exception as e:
        #     logger.error(f"Error converting processed tree to string: {e}")

        # Cách 2: Lưu kết quả vào file SVG mới
        output_filename = "./processed_output_example.svg"
        try:
            processed_tree.write(output_filename, encoding='utf-8', xml_declaration=True)
            logger.info(f"Successfully saved processed SVG to: {output_filename}")
            print(f"Processed SVG saved to: {output_filename}")
        except IOError as e:
            logger.error(f"Error writing processed SVG to file '{output_filename}': {e}")
            print(f"Error saving file: {e}")

    else:
        logger.error("Processing failed. No processed SVG tree was returned.")
        print("Processing failed. Check logs for details.")

    logger.info("--- SvgProcessor Example Finished ---")