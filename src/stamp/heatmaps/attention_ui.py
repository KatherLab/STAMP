import napari
from qtpy.QtGui import QPixmap, QImage
from qtpy.QtWidgets import (
    QComboBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QScrollArea,
    QListWidget,
    QApplication,
    QSlider,
    QHBoxLayout,
    QFrame,
    QLineEdit,
    QFileDialog,
)
from qtpy.QtCore import Qt

import torch
import time
from torch import Tensor
from torch._prims_common import DeviceLikeType
from jaxtyping import Float, Integer
from typing import cast
from collections.abc import Iterable
from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import openslide
import h5py
from scipy.spatial.distance import cdist
from typing import Union
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.preprocessing.tiling import Microns, SlideMPP, SlidePixels, get_slide_mpp_
from stamp.modeling.data import get_coords, get_stride

# Define a generic integer type that can be either Python int or numpy int64
IntType = Union[int, np.int64, np.int32]

__author__ = "Dennis Eschweiler"
__copyright__ = "Copyright (C) 2025 Dennis Eschweiler"
__license__ = "MIT"


def _vals_to_im(
    scores: Float[Tensor, "tile feat"],
    coords_norm: Integer[Tensor, "tile coord"],
) -> Float[Tensor, "width height category"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = coords_norm.max(0).values.flip(0) + 1
    im = torch.ones((*size.tolist(), *scores.shape[1:])).type_as(scores) * -1e-8

    flattened_im = im.flatten(end_dim=1)
    flattened_coords = coords_norm[:, 1] * im.shape[1] + coords_norm[:, 0]
    flattened_im[flattened_coords] = scores

    im = flattened_im.reshape_as(im)

    return im


def _get_thumb(slide, slide_mpp: SlideMPP) -> np.ndarray:
    """Get thumbnail of the slide at the specified MPP and tile size"""
    # Get the thumbnail image from the slide
    dims_um = np.array(slide.dimensions) * slide_mpp
    thumb = slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))
    thumb = np.array(thumb)
    return thumb / 255


def _patch_to_pixmap(patch_image):
    """Convert patch image to QPixmap for display in QLabel using NumPy"""

    # Resize for better visualization if needed
    display_size = 200  # Adjust this value as needed
    aspect_ratio = patch_image.width / patch_image.height

    if aspect_ratio > 1:
        # Wider than tall
        new_width = display_size
        new_height = int(display_size / aspect_ratio)
    else:
        # Taller than wide
        new_height = display_size
        new_width = int(display_size * aspect_ratio)

    patch_image = patch_image.resize((new_width, new_height))

    # Convert PIL image to numpy array
    img_array = np.array(patch_image)

    # Create QImage from NumPy array
    height, width, channels = img_array.shape
    bytes_per_line = channels * width

    # Create QImage (RGB format)
    q_image = QImage(
        img_array.data, width, height, bytes_per_line, QImage.Format_RGB888
    )

    return QPixmap.fromImage(q_image)


class AttentionViewer:
    def __init__(
        self,
        feature_dir: Path,
        wsis_to_process: Iterable[str],
        checkpoint_path: Path,
        output_dir: Path,
        slide_paths: Iterable[Path] | None,
        device: DeviceLikeType,
        default_slide_mpp: SlideMPP | None,
    ):
        """
        Interactive viewer for images with click-based heatmap generation.

        Parameters:
        -----------
        image : np.ndarray
            The base image to display and interact with
        _heatmap_generator : callable, optional
            A function that takes coordinates (y, x) and returns a heatmap
            If None, a simple gaussian will be used
        """
        self.feature_dir = feature_dir
        self.wsis_to_process = wsis_to_process
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.slide_paths = slide_paths
        self.device = device
        self.default_slide_mpp = default_slide_mpp

        # Initialize model
        self.model = (
            LitVisionTransformer.load_from_checkpoint(checkpoint_path).to(device).eval()
        )

        # Initialize napari viewer
        self.viewer = napari.Viewer(title=" Histopathology Attention Explorer")

        # Create dummy image layer
        self.image = np.zeros(
            (100, 100, 3), dtype=np.float32
        )  # Placeholder for the image
        self.image_layer = self.viewer.add_image(self.image, name="Image")
        self.height, self.width = self.image.shape[0], self.image.shape[1]

        # Initialize other attributes
        self.slide = None
        self.attention_map = np.zeros(
            (100, 100), dtype=np.float32
        )  # Placeholder for the attention map
        self.attention_weights = None
        self.token_attn = None
        self._attention_update_debounce = 500
        self._last_attention_update_time = 0
        self.map_coords = None
        self.coords_tile_slide_px = None
        self.tile_size_slide_px = None
        self.selected_token_idx = None
        self.selected_filename = None
        self.num_layer = 0
        self.n_layers = 2
        self.num_head = 0
        self.n_heads = 8

        # Initialize empty heatmap
        self.heatmap = np.zeros((self.height, self.width, 4), dtype=float)
        self.heatmap_layer = self.viewer.add_image(
            self.heatmap, name="Attention", rgb=True, visible=True, opacity=0.7
        )

        # Initialize empty highlight map
        self.highlight_mask = np.zeros((self.height, self.width, 4), dtype=float)
        self.highlight_layer = self.viewer.add_image(
            self.highlight_mask,
            name="Top-k Highlight",
            rgb=True,
            visible=True,
            opacity=1.0,
        )

        # Initialize points layer
        self.clicked_points = []
        self.points_layer = self.viewer.add_points(
            name="Selected Point",
            size=10,
            face_color="green",
            symbol="x",
            n_dimensional=True,
        )
        self._last_processed_point_count = 0
        self._updating_points = False
        self.points_layer.events.data.connect(self._on_add_point)

        # Add other UI elements
        self._add_file_selection_ui()
        self._add_config_selection_ui()
        self._add_topk_controls_ui()
        self._add_patch_display_widget()

        # Disable UI elements until a file is selected
        self._set_ui_enabled(False)

        # Print instructions
        print("Click on the image to select points and generate attention heatmaps")

    #### ADDING UI ELEMENTS ####

    def _add_file_selection_ui(self):
        """Add file selection dropdown and process button"""
        # Create a widget container
        file_selection_widget = QWidget()
        layout = QVBoxLayout()
        file_selection_widget.setLayout(layout)

        # Add a label
        label = QLabel("Available files:")
        layout.addWidget(label)

        # Create a list widget for file selection (scrollable)
        self.file_list = QListWidget()
        self.file_list.addItems(self.wsis_to_process)

        # Set a reasonable height to show multiple items
        self.file_list.setMinimumHeight(150)

        # Make it scrollable if many items
        self.file_list.setVerticalScrollBarPolicy(1)  # 1 = Always show scrollbar

        # Add the list to the layout
        layout.addWidget(self.file_list)

        # Create a process button
        self.process_button = QPushButton("Process Selected File")
        self.process_button.clicked.connect(self._on_process_file)
        layout.addWidget(self.process_button)

        # Add the widget to napari viewer as a dock widget
        self.viewer.window.add_dock_widget(
            file_selection_widget, name="File Selection", area="right"
        )

    def _add_config_selection_ui(self):
        """Add UI controls for selecting attention layer and head"""
        # Create a widget container
        selection_widget = QWidget()
        layout = QVBoxLayout()
        selection_widget.setLayout(layout)

        # Replace the direction slider section in _add_layer_head_selection_ui with:

        # === DIRECTION SELECTION ===
        direction_label = QLabel("Attention Handling:")
        layout.addWidget(direction_label)

        # Create direction selection dropdown
        direction_layout = QHBoxLayout()

        # Direction dropdown
        self.attention_handling = QComboBox()
        self.attention_handling.addItem("From selected tile to others", 0)
        self.attention_handling.addItem("From other tiles to selected", 1)
        self.attention_handling.addItem("Deviation of overall given attention", 2)
        self.attention_handling.addItem("Deviation of overall received attention", 3)
        self.attention_handling.addItem("Mean of overall given attention", 4)
        self.attention_handling.addItem("Mean of overall received attention", 5)
        self.attention_handling.addItem("Class token attention", 6)
        self.attention_handling.addItem("Mutual attention", 7)
        self.attention_handling.addItem("Max mutual attention", 8)

        # Connect the dropdown to the update function
        self.attention_handling.currentIndexChanged.connect(
            self._on_update_attention_map
        )

        direction_layout.addWidget(self.attention_handling)
        layout.addLayout(direction_layout)

        # === LAYER SELECTION ===
        layer_label = QLabel("Number of Network Layer\n(first->last):")
        layout.addWidget(layer_label)

        # Create layer selection controls with arrows and slider
        layer_layout = QHBoxLayout()

        # Layer slider
        self.layer_slider = QSlider(Qt.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(self.n_layers-1)  # Will be updated with actual layer count
        self.layer_slider.setValue(0)
        self.layer_slider.valueChanged.connect(
            lambda value: (
                self.layer_value_label.setText(str(value+1) + f"/{self.n_layers}"),
                self._on_update_attention_map(),
            )
        )
        layer_layout.addWidget(self.layer_slider)

        # Left arrow button for layer
        self.layer_left_btn = QPushButton("←")
        self.layer_left_btn.setMaximumWidth(30)
        self.layer_left_btn.clicked.connect(
            lambda: self._adjust_slider(value=-1, ui_element=self.layer_slider)
        )
        layer_layout.addWidget(self.layer_left_btn)

        # Right arrow button for layer
        self.layer_right_btn = QPushButton("→")
        self.layer_right_btn.setMaximumWidth(30)
        self.layer_right_btn.clicked.connect(
            lambda: self._adjust_slider(value=1, ui_element=self.layer_slider)
        )
        layer_layout.addWidget(self.layer_right_btn)

        # Layer value display
        self.layer_value_label = QLabel("1/2")
        self.layer_value_label.setMinimumWidth(25)
        self.layer_value_label.setAlignment(Qt.AlignCenter)
        layer_layout.addWidget(self.layer_value_label)

        layout.addLayout(layer_layout)

        # === HEAD SELECTION ===
        head_label = QLabel("Number of Prediction Head\n(0 for average):")
        layout.addWidget(head_label)

        # Create head selection controls with arrows and slider
        head_layout = QHBoxLayout()

        # Head slider
        self.head_slider = QSlider(Qt.Horizontal)
        self.head_slider.setMinimum(-1)
        self.head_slider.setMaximum(self.n_heads-1)  # Will be updated with actual head count
        self.head_slider.setValue(-1)
        self.head_slider.valueChanged.connect(
            lambda value: (
                self.head_value_label.setText(str(value+1) + f"/{self.n_heads}"),
                self._on_update_attention_map(),
            )
        )
        head_layout.addWidget(self.head_slider)

        # Left arrow button for head
        self.head_left_btn = QPushButton("←")
        self.head_left_btn.setMaximumWidth(30)
        self.head_left_btn.clicked.connect(
            lambda: self._adjust_slider(-1, self.head_slider)
        )
        head_layout.addWidget(self.head_left_btn)

        # Right arrow button for head
        self.head_right_btn = QPushButton("→")
        self.head_right_btn.setMaximumWidth(30)
        self.head_right_btn.clicked.connect(
            lambda: self._adjust_slider(1, self.head_slider)
        )
        head_layout.addWidget(self.head_right_btn)

        # Head value display
        self.head_value_label = QLabel("1/8")
        self.head_value_label.setMinimumWidth(25)
        self.head_value_label.setAlignment(Qt.AlignCenter)
        head_layout.addWidget(self.head_value_label)

        layout.addLayout(head_layout)

        # === Add the widget to napari viewer as a dock widget ===
        self.viewer.window.add_dock_widget(
            selection_widget, name="Attention Parameters", area="right"
        )

    def _add_topk_controls_ui(self):
        """Add UI controls for top-k tile selection and patch operations"""
        # Create a widget container
        topk_widget = QWidget()
        layout = QVBoxLayout()
        topk_widget.setLayout(layout)

        # === Top-k SELECTION ===
        topk_label = QLabel("Top-k tiles to highlight:")
        layout.addWidget(topk_label)

        # Create top-k selection controls with arrows and slider
        topk_layout = QHBoxLayout()

        # Top-k slider
        self.topk_slider = QSlider(Qt.Horizontal)
        self.topk_slider.setMinimum(0)
        self.topk_slider.setMaximum(50)
        self.topk_slider.setValue(5)
        self.topk_slider.valueChanged.connect(
            lambda value: (
                self.topk_value_label.setText(str(value)),
                self._on_update_attention_map(),
            )
        )
        topk_layout.addWidget(self.topk_slider)

        # Left arrow button for top-k
        self.topk_left_btn = QPushButton("←")
        self.topk_left_btn.setMaximumWidth(30)
        self.topk_left_btn.clicked.connect(
            lambda: self._adjust_slider(-1, self.topk_slider)
        )
        topk_layout.addWidget(self.topk_left_btn)

        # Right arrow button for top-k
        self.topk_right_btn = QPushButton("→")
        self.topk_right_btn.setMaximumWidth(30)
        self.topk_right_btn.clicked.connect(
            lambda: self._adjust_slider(1, self.topk_slider)
        )
        topk_layout.addWidget(self.topk_right_btn)

        # Top-k value display
        self.topk_value_label = QLabel("5")
        self.topk_value_label.setMinimumWidth(25)
        self.topk_value_label.setAlignment(Qt.AlignCenter)
        topk_layout.addWidget(self.topk_value_label)

        layout.addLayout(topk_layout)

        # === LOAD PATCHES BUTTON ===
        self.load_patches_btn = QPushButton("Visualize Top-k Patches")
        self.load_patches_btn.clicked.connect(self._load_tile_patches)
        layout.addWidget(self.load_patches_btn)
        
        save_path_layout = QHBoxLayout()
        
        # Path entry field
        self.save_path_entry = QLineEdit()
        self.save_path_entry.setPlaceholderText("Enter path to save patches...")
        # Set default save path to output directory if available
        if hasattr(self, 'output_dir') and self.output_dir:
            self.save_path_entry.setText(str(Path(self.output_dir) / 'topK_patches'))
        save_path_layout.addWidget(self.save_path_entry)
        
        # Browse button
        self.browse_btn = QPushButton("Browse Save Path")
        self.browse_btn.setMaximumWidth(120)
        self.browse_btn.clicked.connect(self._browse_save_path)
        save_path_layout.addWidget(self.browse_btn)
        
        layout.addLayout(save_path_layout)

        # === SAVE TOP-K BUTTON ===
        self.save_topk_btn = QPushButton("Save Top-k Patches")
        self.save_topk_btn.clicked.connect(self._save_topk_patches)
        layout.addWidget(self.save_topk_btn)

        # === Add the widget to napari viewer as a dock widget ===
        self.viewer.window.add_dock_widget(
            topk_widget, name="Top-k Tile Controls", area="right"
        )

    def _add_patch_display_widget(self):
        """Create a widget to display the selected patch and top-k patches"""
        # Create main widget
        self.patch_display_widget = QWidget()
        layout = QHBoxLayout()
        self.patch_display_widget.setLayout(layout)

        # Create a scroll area to contain patches
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create container widget for patches
        self.patches_container = QWidget()
        self.patches_layout = QHBoxLayout()
        self.patches_container.setLayout(self.patches_layout)

        # Add to scroll area
        scroll_area.setWidget(self.patches_container)
        layout.addWidget(scroll_area)

        # Add the widget to napari viewer as a dock widget at the bottom
        self.viewer.window.add_dock_widget(
            self.patch_display_widget, name="Tile Patches", area="bottom"
        )

    ### UI HANDLING ###

    def _set_ui_enabled(self, enabled: bool):
        """Enable or disable all UI elements"""

        # Enable points layer
        if hasattr(self, "points_layer"):
            self.points_layer.editable = enabled

        # Enable patch loading button
        if hasattr(self, "load_patches_btn"):
            self.load_patches_btn.setEnabled(enabled)

        # Enable save top-k button
        if hasattr(self, "save_topk_btn"):
            self.save_topk_btn.setEnabled(enabled)

        # Enable save path entry and browse button
        if hasattr(self, "save_path_entry"):
            self.save_path_entry.setEnabled(enabled)
        if hasattr(self, "browse_btn"):
            self.browse_btn.setEnabled(enabled)

        # Enable attention handling dropdown
        if hasattr(self, "attention_handling"):
            self.attention_handling.setEnabled(enabled)

        # Enable all sliders
        for slider_name in ["layer_slider", "head_slider", "topk_slider"]:
            if hasattr(self, slider_name):
                getattr(self, slider_name).setEnabled(enabled)

        # Enable all arrow buttons
        for btn_name in [
            "layer_left_btn",
            "layer_right_btn",
            "head_left_btn",
            "head_right_btn",
            "topk_left_btn",
            "topk_right_btn",
        ]:
            if hasattr(self, btn_name):
                getattr(self, btn_name).setEnabled(enabled)

        # Process UI events to update display
        QApplication.processEvents()

    def _adjust_slider(self, value=1, ui_element=None):
        """Adjust the slider value by a given amount"""
        if ui_element is not None:
            current = ui_element.value()
            if (value > 0 and current < ui_element.maximum()) or (
                value < 0 and current > ui_element.minimum()
            ):
                ui_element.setValue(current + value)

    def _update_viewer_image(self, new_image: np.ndarray):
        """
        Update the viewer image and reset heatmap after loading a new file

        Parameters:
        -----------
        new_image : np.ndarray
            The new image to display in the viewer
        """
        # Update the image layer
        self.image = new_image
        self.image_layer.data = self.image

        # Update dimensions based on the new image
        if len(self.image.shape) == 3 and self.image.shape[2] in [3, 4]:  # RGB or RGBA
            self.height, self.width = self.image.shape[0], self.image.shape[1]
        else:  # Grayscale
            self.height, self.width = self.image.shape

        # Reset the heatmap
        self.heatmap = np.zeros((self.height, self.width, 4), dtype=float)
        self.heatmap_layer.data = self.heatmap

        # Clear any existing points
        self.points_layer.data = np.empty((0, 2))
        self.clicked_points = []
        self._last_processed_point_count = 0

        # Reset selected token
        self.selected_token_idx = None

        # Reset viewer scale and position to fit the new image
        self.viewer.reset_view()

        # Set active layer to points layer and mode to add
        self.viewer.layers.selection.active = self.points_layer
        self.points_layer.mode = "add"

        print("Viewer updated with new image")

    ### FILE PROCESSING ###

    def _on_process_file(self):
        """Handle the Process button click"""
        selected_items = self.file_list.selectedItems()

        if selected_items:
            # Get the selected filename
            self.selected_filename = selected_items[0].text()
            print(f"Selected file: {self.selected_filename}")

            self.process_selected_file(self.selected_filename)
            self.load_selected_attention_map()

        else:
            print("No file selected! Please select a file first.")

    def process_selected_file(self, wsi_path):
        """Load the selected file and the corresponding attention map"""

        # Disable UI controls
        self._set_ui_enabled(False)

        # Use QApplication to process events and update the UI
        QApplication.processEvents()

        try:
            print(f"Processing file: {wsi_path}")

            with torch.no_grad():
                # Load WSI
                wsi_path = Path(wsi_path)
                h5_path = self.feature_dir / wsi_path.with_suffix(".h5").name
                print(f"Creating attention map for {wsi_path.name}")

                self.slide = openslide.open_slide(wsi_path)
                slide_mpp = get_slide_mpp_(
                    self.slide, default_mpp=self.default_slide_mpp
                )
                assert slide_mpp is not None, "could not determine slide MPP"

                with h5py.File(h5_path) as h5:
                    feats = (
                        torch.tensor(
                            h5["feats"][:]  # pyright: ignore[reportIndexIssue]
                        )
                        .float()
                        .to(self.device)
                    )

                    coords_um = get_coords(h5).coords_um
                    if not isinstance(coords_um, torch.Tensor):
                        coords_um = torch.tensor(coords_um, dtype=torch.float32)

                    stride_um = Microns(get_stride(coords_um))

                    self.tile_size_slide_px = SlidePixels(
                            int(round(256 / slide_mpp))
                        )
                    if h5.attrs.get("unit") == "um":
                        for attr_name in ["tile_size_um", "tile_size"]:
                            if attr_name in h5.attrs:
                                self.tile_size_slide_px = SlidePixels(
                                    int(round(cast(float, h5.attrs[attr_name]) / slide_mpp))
                                )
                                break                        

                # grid coordinates, i.e. the top-left most tile is (0, 0), the one to its right (0, 1) etc.
                self.map_coords = (coords_um / stride_um).round().long()

                # coordinates as used by OpenSlide (used to extract top/bottom k tiles)
                self.coords_tile_slide_px = torch.round(coords_um / slide_mpp).long()

                # Score for the entire slide
                self.attention_weights = (
                    self.model.vision_transformer.get_attention_maps(
                        bags=feats.unsqueeze(0),
                        coords=coords_um.unsqueeze(0),
                        mask=torch.zeros(
                            1, len(feats), dtype=torch.bool, device=self.device
                        ),
                    )
                )

                # Determine number of heads and layers and update UI elements
                self.n_layers = len(self.attention_weights)
                self.n_heads = self.attention_weights[0].shape[1]
                self.layer_slider.setMaximum(self.n_layers-1)
                self.head_slider.setMaximum(self.n_heads-1)
                self.layer_value_label.setText(f"{self.layer_slider.value()+1}/{self.n_layers}")
                self.head_value_label.setText(f"{self.head_slider.value()+1}/{self.n_heads}")


                # Get thumbnail of the slide
                self.image = _get_thumb(self.slide, slide_mpp)

                # Update the viewer with the new image
                self._update_viewer_image(self.image)

        finally:
            # Re-enable UI controls
            self._set_ui_enabled(True)

    def load_selected_attention_map(self):
        if self.attention_weights is not None:
            # Get attention weights
            # Choose layer
            self.attention_map = self.attention_weights[
                self.num_layer
            ]  # Shape: [batch, heads, tokens, tokens])
            # Choose head (or average)
            if self.num_head == -1:
                # Average over heads
                self.attention_map = self.attention_map.mean(dim=1)
            else:
                self.attention_map = self.attention_map[
                    :, self.num_head, ...
                ]  # Shape: [batch, tokens, tokens]
            # Cut out batch dimension
            self.attention_map = self.attention_map[0, ...]  # Shape: [tokens, tokens]

            # Take absolute values to account positive and negative attention similarly
            self.attention_map = self.attention_map.abs()

            # Normalize attention map to [0, 1] by using percentiles (not considering cls token)
            percentile_low = np.percentile(self.attention_map[1:, 1:], 0.5)
            percentile_high = np.percentile(self.attention_map[1:, 1:], 99.5)
            self.attention_map = (self.attention_map - percentile_low) / (
                percentile_high - percentile_low + 1e-8
            )

    def highlight_top_k_tiles(self):
        if (
            self.selected_token_idx is not None
            and self.token_attn is not None
            and self.map_coords is not None
        ):
            # Create a new highlight mask
            k = min(self.topk_slider.value(), len(self.token_attn))
            highlight_mask = np.zeros((self.height, self.width, 4), dtype=float)

            if k > 0:
                # Get top k indices with highest attention
                top_k_values, top_k_indices = torch.topk(self.token_attn, k)

                # For each top tile, add a colored rectangle to the mask
                for i, (score, idx) in enumerate(
                    zip(top_k_values.cpu().numpy(), top_k_indices.cpu().numpy())
                ):
                    # Get tile coordinates
                    x, y = self.map_coords[idx].cpu().numpy()

                    # Convert to image coordinates (scaled by 8)
                    x_img, y_img = x * 8, y * 8
                    tile_size = 8  # Assuming 8x8 pixels per tile

                    # Create rectangular highlight for this tile
                    # Use a different color intensity based on rank (1st is most intense)
                    min_opacity = 0.5
                    intensity = 1.0 - (
                        i * min_opacity / k
                    )  # Decreasing intensity for lower ranks

                    # Define rectangle in the highlight mask
                    y_start, y_end = max(0, y_img), min(self.height, y_img + tile_size)
                    x_start, x_end = max(0, x_img), min(self.width, x_img + tile_size)

                    # Red with alpha based on score
                    highlight_mask[y_start:y_end, x_start:x_end] = [
                        0.0,
                        0.6,
                        0.0,
                        min(min_opacity, intensity + score * 0.3),
                    ]

            self.highlight_mask = highlight_mask
            self.highlight_layer.data = self.highlight_mask

    def _load_tile_patches(self):
        """Load and display the selected patch and top-k patches"""
        if self.selected_token_idx is None:
            print("No token selected. Click on the image first.")
            return

        # Clear previous patches
        while self.patches_layout.count():
            item = self.patches_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Get selected token patch
        selected_patch = self.slide.read_region(
            tuple(self.coords_tile_slide_px[self.selected_token_idx].tolist()),
            0,
            (self.tile_size_slide_px, self.tile_size_slide_px),
        ).convert("RGB")

        # Add layout for selected patch
        selected_frame = QFrame()
        selected_layout = QVBoxLayout()
        selected_frame.setLayout(selected_layout)
        selected_label = QLabel()
    
        # Create label text and selected image
        if  self.attention_handling.currentData() in (2,3, 4, 5, 6, 8): # If heatmap type is agnostic to token selection
            # Create QLabel for Class token
            selected_pixmap = QPixmap(200, 200) # blank image
            selected_pixmap.fill(Qt.transparent)  
            text_label = QLabel(f"Selected: Class Token Attention")

        else:
            # Create QLabel for image
            selected_pixmap = _patch_to_pixmap(selected_patch)
            text_label = QLabel(f"Selected-ID:{self.selected_token_idx}")
            
        # Add selected patch and label to layout
        selected_label.setPixmap(selected_pixmap)
        selected_layout.addWidget(selected_label)
        text_label.setAlignment(Qt.AlignCenter)
        selected_layout.addWidget(text_label)
        self.patches_layout.addWidget(selected_frame)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(2)
        separator.setMinimumWidth(5)
        separator.setStyleSheet("background-color: #888888;")
        self.patches_layout.addWidget(separator)

        # Get top-k patches
        topk = min(self.topk_slider.value(), len(self.token_attn))
        if topk > 0:
            for n, (score, index) in enumerate(zip(*self.token_attn.topk(topk))):
                # Get patch
                patch = self.slide.read_region(
                    tuple(self.coords_tile_slide_px[index].tolist()),
                    0,
                    (self.tile_size_slide_px, self.tile_size_slide_px),
                ).convert("RGB")

                # Create frame with layout
                patch_frame = QFrame()
                patch_layout = QVBoxLayout()
                patch_frame.setLayout(patch_layout)

                # Create QLabel for image
                patch_label = QLabel()
                patch_pixmap = _patch_to_pixmap(patch)
                patch_label.setPixmap(patch_pixmap)

                # Create label text
                text_label = QLabel(f"Top-{n + 1}-ID:{index} (Score:{score:.2f})")
                text_label.setAlignment(Qt.AlignCenter)

                patch_layout.addWidget(patch_label)
                patch_layout.addWidget(text_label)
                self.patches_layout.addWidget(patch_frame)

        # Force update of the layout
        self.patches_container.adjustSize()
        QApplication.processEvents()

    def _browse_save_path(self):
        """Open file dialog to select save directory"""
        current_path = self.save_path_entry.text() or str(Path(self.output_dir) / 'topK_patches') if hasattr(self, 'output_dir') and self.output_dir else ""
        
        directory = QFileDialog.getExistingDirectory(
            None,
            "Select Directory to Save Patches",
            current_path
        )
        
        if directory:
            self.save_path_entry.setText(directory)

    def _save_topk_patches(self):
        """Save the selected patch and top-k patches to the specified directory"""
        if self.selected_token_idx is None:
            print("No token selected. Click on the image first.")
            return

        save_path = self.save_path_entry.text().strip()
        if not save_path:
            print("Please specify a save path.")
            return

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save selected patch
            selected_patch = self.slide.read_region(
                tuple(self.coords_tile_slide_px[self.selected_token_idx].tolist()),
                0,
                (self.tile_size_slide_px, self.tile_size_slide_px),
            ).convert("RGB")

            # Create filename prefix based on slide and attention parameters
            filename_prefix = f"{Path(self.selected_filename).stem}_layer{self.layer_slider.value()}_head{self.head_slider.value()}_mode{self.attention_handling.currentData()}"

            if self.attention_handling.currentData() == 6:  # Class token attention
                # For class token, we want to skip
                filename_prefix = f"{filename_prefix}_cls"
            else:
                filename_prefix = f"{filename_prefix}_token{self.selected_token_idx}"
                selected_filename = save_dir / f"{filename_prefix}.png"
                selected_patch.save(selected_filename)

            # Save top-k patches
            topk = min(self.topk_slider.value(), len(self.token_attn))
            if topk > 0:
                saved_count = 0
                for n, (score, index) in enumerate(zip(*self.token_attn.topk(topk))):
                    # Get patch
                    patch = self.slide.read_region(
                        tuple(self.coords_tile_slide_px[index].tolist()),
                        0,
                        (self.tile_size_slide_px, self.tile_size_slide_px),
                    ).convert("RGB")

                    # Save patch
                    patch_filename = save_dir / f"{filename_prefix}_top{n+1}_token{index}_score{score:.3f}.png"
                    patch.save(patch_filename)
                    saved_count += 1

            print(f"Saved {saved_count} top-k patches to: {save_dir}")

        except Exception as e:
            print(f"Error saving patches: {e}")

    def _on_update_attention_map(self):
        # Check if we have data to display
        if self.attention_weights is None:
            return

        # Simple debounce to avoid too frequent updates
        current_time = time.time() * 1000  # Convert to milliseconds
        if (
            current_time - self._last_attention_update_time
            < self._attention_update_debounce
        ):
            return
        self._last_attention_update_time = current_time

        # Get head and layer values from the sliders
        self.num_layer = self.layer_slider.value()
        self.num_head = self.head_slider.value()

        # Update attention map
        self.load_selected_attention_map()
        self._last_processed_point_count = 0  # Reset last processed point count
        self._on_add_point()

    def _on_add_point(self):
        """Handle points being added to the points layer"""
        if self.map_coords is not None:
            # Prevent recursive calls
            if self._updating_points:
                return

            # Check if points have been added
            if (
                len(self.points_layer.data) > self._last_processed_point_count
            ):  # If there's any data
                # Keep only the last added point
                last_point = self.points_layer.data[-1]

                # Convert to proper types
                y, x = int(last_point[0]), int(last_point[1])

                # Set the flag before updating to prevent recursion
                self._updating_points = True

                try:
                    # Update heatmap based on the new point
                    self.update_heatmap(y - 4, x - 4)  # 4 to center the point

                    # Snap to selected token position
                    x_snapped, y_snapped = self.map_coords[
                        self.selected_token_idx, :
                    ].tolist()
                    self.points_layer.data = np.array(
                        [[y_snapped * 8 + 4, x_snapped * 8 + 4]]
                    )
                    self._last_processed_point_count = 1  # We now have 1 point

                    # Update top-k tiles
                    self.highlight_top_k_tiles()

                    # Print clicked coordinates
                    print(
                        f"Clicked at coordinates: ({y},{x}). Selected token index: {self.selected_token_idx} at ({y_snapped * 8 + 4},{x_snapped * 8 + 4})"
                    )
                finally:
                    # Reset the flag after updating
                    self._updating_points = False

        else:
            print("No map coordinates available. Please load a file first.")

    def update_heatmap(self, y: IntType, x: IntType):
        """Update the heatmap based on clicked position"""
        # Generate new heatmap using the provided or default function
        self.heatmap, self.selected_token_idx = self._heatmap_generator(y, x)

        # Update the heatmap layer
        self.heatmap_layer.data = self.heatmap

    def get_token_attention(self, selected_token_idx: IntType):
        # Get selected direction
        selected_direction = self.attention_handling.currentData()

        # Get attention for selected token

        # Attention from selected to others
        if selected_direction == 0:
            token_attn = self.attention_map[
                selected_token_idx + 1, 1:
            ]  # +1 to skip the cls token

        # Attention from others to selected
        elif selected_direction == 1:
            token_attn = self.attention_map[
                1:, selected_token_idx + 1
            ]  # +1 to skip the cls token

        # Deviation of overall given attention
        elif selected_direction == 2:
            token_attn = torch.std(self.attention_map[1:, 1:], dim=0)
            percentile_low = np.percentile(token_attn, 0.5)
            percentile_high = np.percentile(token_attn, 99.5)
            token_attn = (token_attn - percentile_low) / (
                percentile_high - percentile_low + 1e-8
            )

        # Deviation of overall received attention
        elif selected_direction == 3:
            token_attn = torch.std(self.attention_map[1:, 1:], dim=1)
            percentile_low = np.percentile(token_attn, 0.5)
            percentile_high = np.percentile(token_attn, 99.5)
            token_attn = (token_attn - percentile_low) / (
                percentile_high - percentile_low + 1e-8
            )

        # Mean of overall given attention
        elif selected_direction == 4:
            token_attn = torch.mean(self.attention_map[1:, 1:], dim=0)

        # Mean of overall received attention
        elif selected_direction == 5:
            token_attn = torch.mean(self.attention_map[1:, 1:], dim=1)

        # Class token attention
        elif selected_direction == 6:
            token_attn = self.attention_map[0, 1:]  # from cls to others
            percentile_low = np.percentile(token_attn, 0.5)
            percentile_high = np.percentile(token_attn, 99.5)
            token_attn = (token_attn - percentile_low) / (
                percentile_high - percentile_low + 1e-8
            )

        # Mutual attention
        elif selected_direction == 7:
            mutual_attn_matrix = self.attention_map[1:, 1:] * self.attention_map[1:, 1:].T
            token_attn = mutual_attn_matrix[:,selected_token_idx]

        # Mean mutual attention
        elif selected_direction == 8:
            mutual_attn_matrix = self.attention_map[1:, 1:] * self.attention_map[1:, 1:].T
            token_attn = torch.max(mutual_attn_matrix, dim=1)[0]

        else:
            raise ValueError(f"Invalid direction selected: {selected_direction}")

        token_attn = np.clip(token_attn, 0, 1)

        return token_attn

    def _heatmap_generator(self, y: IntType, x: IntType):
        """Heatmap generator - determines closest token to clicked position and extract inter-token attention"""
        # Get the closest token to the clicked position
        token_distances = cdist(
            [(x, y)], self.map_coords.numpy(force=True) * 8
        )  # Upscale by 8 to match thumbnail size
        selected_token_idx = np.argmin(token_distances)

        # Get attention for selected token
        self.token_attn = self.get_token_attention(selected_token_idx)

        # Generate heatmap
        cls_attn_map = _vals_to_im(
            self.token_attn.unsqueeze(-1),  # Add feature dimension
            self.map_coords,
        ).squeeze(-1)  # Shape: [width, height]

        # Upscale by 8 to match the thumbnail size
        cls_attn_map = cls_attn_map.repeat_interleave(8, dim=0).repeat_interleave(
            8, dim=1
        )

        # Normalize the heatmap to [0, 1]
        # cls_attn_map = (cls_attn_map - cls_attn_map.min()) / (cls_attn_map.max() - cls_attn_map.min() + 1e-8)

        # Convert to numpy array
        heatmap_values = cls_attn_map.numpy(force=True)

        # Get the colormap
        colormap = cm.get_cmap("inferno")

        # Apply colormap to the values to get RGB
        heatmap_rgba = colormap(heatmap_values)

        # Create a mask for zero and near-zero values (make these transparent)
        threshold = 0.0
        zero_mask = heatmap_values < threshold

        # Set alpha channel to make zero-value regions fully transparent
        heatmap_rgba[zero_mask, 3] = 0.0

        # Scale the alpha for non-zero values by the desired opacity
        heatmap_rgba[~zero_mask, 3] *= 1.0

        return heatmap_rgba, selected_token_idx

    def show(self):
        """Display the viewer and start the event loop"""
        napari.run()


def show_attention_ui(
    feature_dir: Path,
    wsis_to_process: Iterable[str],
    checkpoint_path: Path,
    output_dir: Path,
    slide_paths: Iterable[Path] | None,
    device: DeviceLikeType,
    default_slide_mpp: SlideMPP | None,
):
    """
    Launch the attention UI.

    Parameters:
    -----------
    feature_dir : Path
        Directory containing feature files
    wsis_to_process : Iterable[str]
        List of WSI files to present for process
    checkpoint_path : Path
        Path to model checkpoint
    output_dir : Path
        Directory to save output files
    slide_paths : Iterable[Path] | None
        Paths to specific slide files
    device : DeviceLikeType
        Device to run model on
    default_slide_mpp : SlideMPP | None
        Default slide microns per pixel
    """
    AttentionViewer(
        feature_dir,
        wsis_to_process,
        checkpoint_path,
        output_dir,
        slide_paths,
        device,
        default_slide_mpp,
    )
    napari.run()
