#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import argparse
import yaml 
from pathlib import Path
import threading
import time 

# --- Global variables for thread communication ---
g_image_to_display = None
g_window_name = "Image Viewer"
g_request_window_close = False
g_window_active = False
g_display_lock = threading.Lock() # To protect access to shared variables

def dir_path(path_string):
    if os.path.isdir(path_string):
        return path_string
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path_string} is not a valid path or directory")

def display_thread_func():
    """
    This function runs in a separate thread and handles displaying the image.
    It continuously calls waitKey(1) to keep the window responsive.
    """
    global g_image_to_display, g_window_name, g_request_window_close, g_window_active, g_display_lock
    
    # cv2.namedWindow(g_window_name, cv2.WINDOW_AUTOSIZE) # Create window once

    while not g_request_window_close:
        with g_display_lock:
            current_image = g_image_to_display # Get the current image to show
            window_should_be_active = g_window_active

        if window_should_be_active and current_image is not None:
            try:
                # Ensure window exists before showing. namedWindow can be called multiple times.
                cv2.namedWindow(g_window_name, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(g_window_name, current_image)
            except cv2.error as e:
                print(f"OpenCV error in display thread: {e}")
                # Potentially try to recreate window or handle error
                time.sleep(0.1) # Avoid tight loop on error
                continue # Try again
            except Exception as e:
                print(f"Unexpected error in display_thread imshow: {e}")
                time.sleep(0.1)
                continue
        elif not window_should_be_active:
            # If window was active but now requested to be inactive, destroy it
            # This logic might need refinement if window is only hidden not destroyed
            try:
                # Check if window exists before trying to destroy
                if cv2.getWindowProperty(g_window_name, cv2.WND_PROP_VISIBLE) >= 1:
                     cv2.destroyWindow(g_window_name)
            except cv2.error: # Window might already be gone
                pass
            except Exception as e:
                print(f"Unexpected error destroying window: {e}")


        key = cv2.waitKey(30) # Process GUI events, wait 30ms. Essential! [1]
                              # A small delay is crucial. waitKey(1) is common.

        if key != -1: # If a key was pressed in the GUI window
            # print(f"Key pressed in GUI: {key}") # For debugging
            # Implement q in GUI to signal main thread to quit
            if key & 0xFF == ord('q'): # 'q' pressed in GUI window
                with g_display_lock:
                    g_request_window_close = True # Signal main thread to quit
                    g_window_active = False # Also hide window immediately
                print("Quit requested from GUI window.")


        # Check if window was manually closed by user (via 'X' button)
        # This check can be unreliable or behave differently based on OS/OpenCV backend
        try:
            if window_should_be_active and current_image is not None and cv2.getWindowProperty(g_window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("GUI window closed by user (X button).")
                with g_display_lock:
                    g_request_window_close = True # Signal main thread to quit
                    g_window_active = False
                break # Exit display thread
        except cv2.error: # Window might not exist
            pass
        except Exception: # Other potential errors
            pass


    # Final cleanup for the display thread
    try:
        cv2.destroyAllWindows() # Close any remaining OpenCV windows from this thread
    except cv2.error:
        pass # Ignore if no windows or errors
    print("Display thread finished.")


def load_and_prepare_image(image_path, label_path, class_names_list=None, alpha=0.5):
    """
    Loads an image, its labels, and returns the annotated image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None, "Image file not found"
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None, "Could not read image"
        
    img_height, img_width, _ = image.shape
    overlay = image.copy()
    annotated_image = image.copy()

    label_found = False
    labels_content = []
    status_message = ""

    if os.path.exists(label_path):
        label_found = True
        try:
            with open(label_path, 'r') as f:
                labels_content = f.readlines()
        except Exception as e:
            status_message = f"Error reading label file {label_path}: {e}"
            labels_content = []
            label_found = False
    else:
        status_message = "No Label File"

    if label_found:
        if not labels_content:
            status_message = "Empty Label File"
        for i, label_line in enumerate(labels_content):
            try:
                parts = label_line.strip().split()
                if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                    continue
                class_id = int(parts[0])
                polygon_norm = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                polygon_abs = (polygon_norm * np.array([img_width, img_height])).astype(np.int32)
                colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
                    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), 
                    (0, 128, 128), (128, 0, 128), (192, 192, 192), (255,165,0), 
                    (75,0,130), (0,100,0)
                ]
                color = colors[class_id % len(colors)]
                cv2.fillPoly(overlay, [polygon_abs], color)
                cv2.polylines(annotated_image, [polygon_abs], isClosed=True, color=color, thickness=2)
                label_text_display = f"ID: {class_id}"
                if class_names_list and 0 <= class_id < len(class_names_list):
                    if class_names_list[class_id] is not None:
                         label_text_display = str(class_names_list[class_id])
                text_position_x = polygon_abs[0, 0]
                text_position_y = polygon_abs[0, 1] - 10
                if text_position_y < 20 : text_position_y = polygon_abs[0, 1] + 20
                cv2.putText(annotated_image, label_text_display, (text_position_x, text_position_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except ValueError: pass # Malformed numeric
            except Exception: pass # Other errors
                
        if labels_content: # Only blend if there were valid labels
             cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
    
    return annotated_image, status_message


def browse_dataset(images_dir_path, labels_dir_path, class_names_list=None):
    global g_image_to_display, g_window_name, g_request_window_close, g_window_active, g_display_lock

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_files = sorted([f for ext in image_extensions for f in glob.glob(os.path.join(images_dir_path, ext))])

    if not image_files:
        print(f"No images found in directory: {images_dir_path}")
        return

    total_images = len(image_files)
    print_info(images_dir_path, labels_dir_path, total_images, class_names_list)

    current_index = -1

    # Start the display thread
    # cv2.namedWindow(g_window_name, cv2.WINDOW_AUTOSIZE) # Create window once in main thread
    display_thread = threading.Thread(target=display_thread_func, daemon=True)
    display_thread.start()

    # --- Automatically display the first image ---
    if total_images > 0:
        current_index = 0
        update_display(current_index, image_files, labels_dir_path, class_names_list, total_images)

    while not g_request_window_close: # Check global flag
        prompt_text = create_prompt(current_index, total_images)
        try:
            user_input = input(prompt_text).lower().strip()
            if not user_input and current_index != -1: # Just pressing Enter might mean "refresh" or "do nothing"
                continue
            if not user_input and current_index == -1: # No image shown yet, empty input is invalid
                print("Please enter a command.")
                continue

            if user_input == 'q':
                print("Quit requested from command line.")
                with g_display_lock:
                    g_request_window_close = True
                    g_window_active = False # Hide window
                break 
            
            target_index = -1
            if user_input == 'n':
                if current_index < total_images - 1:
                    target_index = current_index + 1
                else:
                    print("Already at the last image.")
                    continue
            elif user_input == 'p':
                if current_index > 0:
                    target_index = current_index - 1
                else:
                    print("Already at the first image.")
                    continue
            else:
                order_number = int(user_input)
                if 1 <= order_number <= total_images:
                    target_index = order_number - 1
                else:
                    print(f"Invalid number. Please enter a number between 1 and {total_images}.")
                    continue
            
            if target_index != -1:
                current_index = target_index
                update_display(current_index, image_files, labels_dir_path, class_names_list, total_images)
            
        except ValueError:
            if user_input: # Avoid error message for empty input if already handled
                print("Invalid input. Please enter a number, 'n', 'p', or 'q'.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user (Ctrl+C).")
            with g_display_lock:
                g_request_window_close = True
                g_window_active = False
            break
        except Exception as e:
            print(f"An error occurred in the browsing loop: {e}")
            import traceback
            traceback.print_exc()
            # Decide to break or continue
            with g_display_lock: # Ensure thread is signaled on unexpected error
                g_request_window_close = True
                g_window_active = False
            break

    print("Waiting for display thread to finish...")
    if display_thread.is_alive():
        display_thread.join(timeout=2.0) # Wait for thread to exit
    if display_thread.is_alive():
        print("Warning: Display thread did not exit cleanly.")
    
    print("Image browser for this split finished.")
    # cv2.destroyAllWindows() should be handled by display_thread or final cleanup if needed

def update_display(index, image_files, labels_dir_path, class_names_list, total_images):
    """Helper to load and signal the display thread to update the image."""
    global g_image_to_display, g_window_active, g_display_lock
    
    image_path = image_files[index]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(labels_dir_path, base_name + ".txt")
    
    print(f"\nLoading image {index + 1}/{total_images}: {image_path}")
    
    annotated_img, status_msg = load_and_prepare_image(image_path, label_path, class_names_list)
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found at {label_path}")
    elif status_msg and "Empty" in status_msg:
        print(f"Info: {status_msg}")


    with g_display_lock:
        if annotated_img is not None:
            g_image_to_display = annotated_img
            g_window_active = True
            # Update window title dynamically if possible (can be tricky with threaded imshow)
            # For simplicity, keeping g_window_name static or update it carefully if needed.
            # cv2.setWindowTitle(g_window_name, f"Image: {os.path.basename(image_path)} ({status_msg})") # This might need to be in display_thread
        else:
            print(f"Failed to load or prepare image: {status_msg if status_msg else 'Unknown error'}")
            # Optionally display a placeholder or clear the window
            g_image_to_display = np.zeros((480, 640, 3), dtype=np.uint8) # Black image
            cv2.putText(g_image_to_display, "Error loading image", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            g_window_active = True # Still show the error image

def print_info(images_dir, labels_dir, total_images, class_names):
    print(f"\nFound {total_images} images in '{images_dir}'.")
    print(f"Corresponding labels are expected in '{labels_dir}'.")
    if class_names:
        actual_names = sum(1 for name in class_names if name is not None)
        print(f"Loaded {actual_names} class names (list size {len(class_names)} for sparse IDs).")
    else:
        print("No class names loaded; will display class IDs.")

def create_prompt(current_idx, total_imgs):
    if current_idx != -1:
        next_s = f"{current_idx + 2}/{total_imgs}" if current_idx + 1 < total_imgs else "end"
        prev_s = f"{current_idx}/{total_imgs}" if current_idx > 0 else "start"
        return (f"\nCmd (img #, 'n' ({next_s}), 'p' ({prev_s}), 'q' to quit): ")
    return f"\nCmd (img # (1-{total_imgs}), 'q' to quit): "


# --- YAML Parsing Function (from previous response, ensure it's included) ---
def parse_dataset_yaml(yaml_file_path, dataset_root_dir_path):
    config = {'names': None, 'splits': {}}
    try:
        with open(yaml_file_path, 'r') as f:
            data = yaml.safe_load(f)

        yaml_base_path_segment = data.get('path', '.')
        effective_base_path = (Path(dataset_root_dir_path) / yaml_base_path_segment).resolve()

        if 'names' in data and isinstance(data['names'], (dict, list)):
            if isinstance(data['names'], dict):
                names_dict = data['names']
                max_idx = -1
                valid_names_dict = {}
                for k, v in names_dict.items():
                    if isinstance(k, int) and k >= 0:
                        max_idx = max(max_idx, k)
                        valid_names_dict[k] = str(v)
                if max_idx != -1:
                    class_names_list = [None] * (max_idx + 1)
                    for idx_val, name_val in valid_names_dict.items():
                        class_names_list[idx_val] = name_val
                    config['names'] = class_names_list
                elif not valid_names_dict and names_dict:
                     config['names'] = [str(v) for v in names_dict.values()]
                else: pass # Warning handled if names is still None
            elif isinstance(data['names'], list):
                config['names'] = [str(name) for name in data['names']]
        
        if config.get('names') is None: # If still None after trying
            print(f"Warning: 'names' field not found, invalid, or empty in {yaml_file_path}.")


        possible_split_keys = ['train', 'val', 'test'] + [k for k in data if k not in ['path', 'names', 'nc', 'download']]
        for split_name in possible_split_keys:
            if split_name in data and data[split_name] and isinstance(data[split_name], str):
                relative_image_dir_segment = data[split_name]
                abs_image_path = (effective_base_path / relative_image_dir_segment).resolve()
                
                path_parts_from_yaml = list(Path(relative_image_dir_segment).parts)
                labels_path_parts_yaml = []
                yaml_images_replaced = False
                for part in path_parts_from_yaml:
                    if part.lower() == "images":
                        labels_path_parts_yaml.append("labels")
                        yaml_images_replaced = True
                    else:
                        labels_path_parts_yaml.append(part)
                
                abs_label_path = (effective_base_path / Path(*labels_path_parts_yaml)).resolve()

                if not yaml_images_replaced and "labels" not in [p.lower() for p in labels_path_parts_yaml]:
                     print(f"Warning: Segment 'images' not found in YAML path '{relative_image_dir_segment}' for split '{split_name}'. "
                           f"Label path '{abs_label_path}' was inferred by replacing 'images' with 'labels' in base path structure. Verify correctness.")


                if os.path.isdir(abs_image_path) and os.path.isdir(abs_label_path):
                    config['splits'][split_name] = {
                        'images': str(abs_image_path),
                        'labels': str(abs_label_path)
                    }
                else:
                    if not os.path.isdir(abs_image_path):
                        print(f"Error: Image directory for split '{split_name}' not found: {abs_image_path}")
                    if not os.path.isdir(abs_label_path):
                         print(f"Error: Label directory for split '{split_name}' not found: {abs_label_path}")
        return config
    except FileNotFoundError:
        print(f"Error: YAML configuration file not found at {yaml_file_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {yaml_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {yaml_file_path}: {e}")
        import traceback
        traceback.print_exc()
    return None

# --- Main Execution (from previous response, ensure it's included) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Browse and visualize YOLO segmentation dataset images with their labels, using a dataset.yaml file for configuration.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=dir_path,
        required=True,
        help="Path to the root directory of the dataset. \n"
             "This directory should contain a 'dataset.yaml' or 'data.yaml' file, \n"
             "and subdirectories for images and labels as defined in the YAML."
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="(Optional) Specify which dataset split to browse (e.g., 'train', 'val', 'test'). \n"
             "If not provided, defaults to 'train' if available, then 'val', otherwise prompts."
    )

    args = parser.parse_args()

    yaml_path = None
    possible_yaml_names = ["dataset.yaml", "data.yaml"]
    for name in possible_yaml_names:
        path = os.path.join(args.dataset_root_dir, name)
        if os.path.isfile(path):
            yaml_path = path
            break
    
    if not yaml_path:
        print(f"Error: Could not find 'dataset.yaml' or 'data.yaml' in '{args.dataset_root_dir}'.")
        exit(1)
    
    print(f"Using dataset configuration file: {yaml_path}")
    dataset_config = parse_dataset_yaml(yaml_path, args.dataset_root_dir)

    if not dataset_config or not dataset_config.get('splits'):
        print("Error: Failed to load or parse dataset configuration, or no valid splits found.")
        exit(1)

    available_splits = list(dataset_config['splits'].keys())
    if not available_splits:
        print("Error: No valid data splits found in the dataset configuration.")
        exit(1)
    
    chosen_split_name = None
    if args.split:
        if args.split in available_splits:
            chosen_split_name = args.split
        else:
            print(f"Error: Specified split '{args.split}' is not available. Available splits: {', '.join(available_splits)}")
            exit(1)
    else:
        if 'train' in available_splits: chosen_split_name = 'train'; print(f"Defaulting to 'train' split.")
        elif 'val' in available_splits: chosen_split_name = 'val'; print(f"Defaulting to 'val' split (train not found).")
        elif len(available_splits) == 1: chosen_split_name = available_splits[0]; print(f"Automatically selected only available split: '{chosen_split_name}'")
        else:
            print("\nAvailable dataset splits:"); [print(f"  {i+1}. {name}") for i, name in enumerate(available_splits)]
            while True:
                try:
                    choice_idx = int(input(f"Please choose a split number (1-{len(available_splits)}): ")) - 1
                    if 0 <= choice_idx < len(available_splits): chosen_split_name = available_splits[choice_idx]; break
                    else: print("Invalid choice.")
                except ValueError: print("Invalid input.")
                except KeyboardInterrupt: print("\nOperation cancelled."); exit(0)

    if not chosen_split_name: print("Error: No split was selected."); exit(1)
        
    print(f"\nSelected split: '{chosen_split_name}'")
    
    selected_split_data = dataset_config['splits'][chosen_split_name]
    browse_dataset(selected_split_data['images'], selected_split_data['labels'], dataset_config.get('names'))

    print("\nScript finished.")
