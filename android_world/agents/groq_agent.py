# standards
import json
import os
import re
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Literal, Optional, Tuple, Union

# third-party
import torch
import json_repair
from groq import Groq
from pydantic import BaseModel, Field, TypeAdapter

# local
from android_env.proto.a11y import android_accessibility_forest_pb2
from android_env.proto.a11y.android_accessibility_node_info_pb2 import (
    AndroidAccessibilityNodeInfo,
)
from android_world.agents.base_agent import (
    AgentInteractionResult,
    EnvironmentInteractingAgent,
)
from android_world.env import interface, json_action


def parse_json(text):
    try:
        return json_repair.loads(text)
    except:
        return None


# action Space
class ClickAction(BaseModel):
    action_type: Literal["click"] = "click"
    element_id: str = Field(..., description="UI element ID to click")


class LongPressAction(BaseModel):
    action_type: Literal["long_press"] = "long_press"
    element_id: str = Field(..., description="UI element ID to long press")


class ScrollAction(BaseModel):
    action_type: Literal["scroll"] = "scroll"
    direction: Literal["up", "down", "left", "right"] = Field(
        ..., description="Scroll direction"
    )


class InputTextAction(BaseModel):
    action_type: Literal["input_text"] = "input_text"
    text: str = Field(..., description="Text to input")


class OpenAppAction(BaseModel):
    action_type: Literal["open_app"] = "open_app"
    app_name: str = Field(..., description="App name to open")


class NavigateHomeAction(BaseModel):
    action_type: Literal["navigate_home"] = "navigate_home"


class NavigateBackAction(BaseModel):
    action_type: Literal["navigate_back"] = "navigate_back"


class WaitAction(BaseModel):
    action_type: Literal["wait"] = "wait"


class DoneAction(BaseModel):
    action_type: Literal["done"] = "done"


UIAction = Union[
    ClickAction,
    LongPressAction,
    ScrollAction,
    InputTextAction,
    OpenAppAction,
    NavigateHomeAction,
    NavigateBackAction,
    WaitAction,
    DoneAction,
]

UIActionAdapter = TypeAdapter(UIAction)
action_space_schema = UIActionAdapter.json_schema()


@dataclass
class View:
    """compressed UI view element"""

    tag: str
    text: str = ""
    unique_id: str = ""
    resource_id: str = ""
    content_desc: str = ""
    clickable: bool = False
    scrollable: bool = False
    checkable: bool = False
    bounds: str = ""
    children: List["View"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def to_string(self, indent: int = 0) -> str:
        prefix = " " * indent
        attrs = []

        if self.text:
            attrs.append(f"text: {self.text}")

        if self.content_desc:
            attrs.append(f"desc: {self.content_desc}")

        if self.resource_id:
            # remove package prefix
            short_id = (
                self.resource_id.split("/")[-1]
                if "/" in self.resource_id
                else self.resource_id
            )
            attrs.append(f"id: {short_id}")

        # interaction flags
        flags = []
        if self.clickable:
            flags.append("clickable")
        if self.scrollable:
            flags.append("scrollable")
        if self.checkable:
            flags.append("checkable")

        attr_str = "; ".join(attrs) if attrs else ""
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        # example: " [ui_1] ViewGroup (id: content)""
        # [CONTAINS THE ID INSTEAD OF DEPTH]

        result = f"{prefix}[{self.unique_id}] {self.tag}"
        if attr_str:
            result += f" ({attr_str} {flag_str})" if flag_str else f" ({attr_str})"
        elif flag_str:
            result += f" {flag_str}"
        result += "\n"

        # add children
        for child in self.children:
            result += child.to_string(indent + 1)

        return result


class UITreeCompressor:
    # non-semantic container types that can often be merged
    CONTAINER_TYPES = {
        "android.widget.LinearLayout",
        "android.widget.RelativeLayout",
        "android.widget.FrameLayout",
        "android.view.ViewGroup",
        "android.widget.ScrollView",
        "android.widget.HorizontalScrollView",
    }

    # semantic types that should be preserved
    SEMANTIC_TYPES = {
        "android.widget.Button",
        "android.widget.ImageButton",
        "android.widget.EditText",
        "android.widget.TextView",
        "android.widget.CheckBox",
        "android.widget.RadioButton",
        "android.widget.Switch",
        "android.widget.ImageView",
        "android.widget.ListView",
        "android.widget.RecyclerView",
    }

    TAG_MAPPING = {
        # semantic
        "Button": "button",
        "ImageButton": "button",
        "EditText": "input",
        "TextView": "text",
        "CheckBox": "checkbox",
        "RadioButton": "radio",
        "Switch": "toggle",
        "ImageView": "icon",
        "ListView": "list",
        "RecyclerView": "list",
        # containers
        "LinearLayout": "container",
        "RelativeLayout": "container",
        "FrameLayout": "container",
        "ViewGroup": "container",
        "ScrollView": "scroll",
        "HorizontalScrollView": "scroll",
        "CardView": "card",
        "LinearLayoutCompat": "container",
    }

    def __init__(self):
        self.original_nodes = 0
        self.compressed_nodes = 0
        self.unique_id_counter = 0
        self.id_to_bounds = {}

    # to use with the emulator
    def compress_xml_string(self, xml_string: str) -> str:
        try:
            self.original_nodes = 0
            self.compressed_nodes = 0
            self.unique_id_counter = 0
            self.id_to_bounds = {}

            root = ET.fromstring(xml_string)
            self.original_nodes = self._count_nodes(root)

            compressed_views = self._transform_xml_node(root, [])
            self.compressed_nodes = self._count_views(compressed_views)

            result = ""
            for view in compressed_views:
                result += view.to_string()

            return result.strip(), self.id_to_bounds
        except ET.ParseError as e:
            return f"Error parsing XML: {e}\n{xml_string}"

    def compress_xml_file(self, xml_path: str) -> str:
        with open(xml_path, "r", encoding="utf-8") as f:
            return self.compress_xml_string(f.read())

    # to use with the dataset
    def compress_protobuf(self, serialized):
        nodes = self._parse_forest_proto(serialized)

        id_to_node = {n.unique_id: n for n in nodes}

        all_nodes_bounds = {}
        for node in nodes:
            if node.is_visible_to_user:
                bounds = f"[{node.bounds_in_screen.left},{node.bounds_in_screen.top}][{node.bounds_in_screen.right},{node.bounds_in_screen.bottom}]"
                all_nodes_bounds[f"original_{node.unique_id}"] = bounds

        # find roots
        root_nodes = [node for node in nodes if node.depth == 0]

        # process each root
        compressed_views = []
        for root in root_nodes:
            views = self._transform_proto_node(root, id_to_node, set())
            compressed_views.extend(views)

        result = ""
        for view in compressed_views:
            result += view.to_string()

        return result, self.id_to_bounds, all_nodes_bounds

    def _parse_forest_proto(self, serialized):
        self.original_nodes = 0
        self.compressed_nodes = 0
        self.unique_id_counter = 0
        self.id_to_bounds = {}

        forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
        forest.ParseFromString(serialized)

        nodes_out = []

        for window in forest.windows:
            tree = window.tree

            for node in tree.nodes:
                nodes_out.append(node)
                node.is_clickable

        return nodes_out

    def _transform_xml_node(
        self, node: ET.Element, child_views: List[View]
    ) -> List[View]:
        """Transform xml node recursively process children first (bottom-up approach)"""

        processed_children = []
        for child in node:
            child_result = self._transform_xml_node(child, [])
            processed_children.extend(child_result)

        # leaf node
        if len(processed_children) == 0:
            return self._handle_leaf_xml_node(node)

        # filter node?
        if self._should_filter_xml_node(node):
            return processed_children

        # merge?
        if self._should_merge_xml_with_children(node, processed_children):
            merged_view = self._merge_xml_node_with_children(node, processed_children)
            return [merged_view]

        # pass through this node
        view = self._create_view_from_xml(node)
        view.children = processed_children
        return [view]

    def _transform_proto_node(
        self,
        node: AndroidAccessibilityNodeInfo,
        id_to_node: Dict[int, AndroidAccessibilityNodeInfo],
        visited: set,
    ) -> List[View]:
        """Transform protobuf node recursively process children first (bottom-up approach)"""

        # avoid cycles
        if node.unique_id in visited:
            return []
        visited.add(node.unique_id)

        # process children first
        processed_children = []
        for child_id in node.child_ids:
            child_node = id_to_node.get(child_id)
            if child_node:
                child_views = self._transform_proto_node(
                    child_node, id_to_node, visited
                )
                processed_children.extend(child_views)

        # leaf node
        if len(processed_children) == 0:
            return self._handle_leaf_proto_node(node)

        # filter node?
        if self._should_filter_proto_node(node):
            return processed_children

        # merge?
        if self._should_merge_proto_with_children(node, processed_children):
            merged = self._merge_proto_node_with_children(node, processed_children)
            return [merged]

        # oass through this node
        view = self._create_view_from_proto(node)
        view.children = processed_children
        return [view]

    def _handle_leaf_xml_node(self, node: ET.Element) -> List[View]:
        if self._is_leaf_filtered_xml(node):
            return []

        # it's a semantic node, use it
        view = self._create_view_from_xml(node)
        return [view]

    def _handle_leaf_proto_node(self, node: AndroidAccessibilityNodeInfo) -> List[View]:
        if self._is_leaf_filtered_proto(node):
            return []

        # it's a semantic node, use it
        view = self._create_view_from_proto(node)
        return [view]

    def _should_filter_xml_node(self, node: ET.Element) -> bool:
        """Determine if xml node should be filtered out"""

        # filter if not visible
        if not self._is_visible_xml(node):
            return True

        attrs = node.attrib
        has_text = bool(attrs.get("text", "").strip())
        has_content_desc = bool(attrs.get("content-desc", "").strip())
        has_resource_id = bool(attrs.get("resource-id", "").strip())
        is_interactive = self._is_interactive_xml(node)

        class_name = attrs.get("class", "")
        is_generic_container = class_name in self.CONTAINER_TYPES

        # filter containers with no semantic value
        if is_generic_container and not (
            has_text or has_content_desc or has_resource_id or is_interactive
        ):
            return True

        return False

    def _should_filter_proto_node(self, node: AndroidAccessibilityNodeInfo) -> bool:
        """Determine if protobuf node should be filtered out"""

        # filter if not visible
        if not self._is_visible_proto(node):
            return True

        has_text = bool((node.text or "").strip())
        has_content_desc = bool((node.content_description or "").strip())
        has_resource_id = bool((node.view_id_resource_name or "").strip())
        is_interactive = self._is_interactive_proto(node)

        class_name = node.class_name or ""
        is_generic_container = class_name in self.CONTAINER_TYPES

        # filter containers with no semantic value
        if is_generic_container and not (
            has_text or has_content_desc or has_resource_id or is_interactive
        ):
            return True

        return False

    def _should_merge_xml_with_children(
        self, node: ET.Element, children: List[View]
    ) -> bool:
        """Determine if xml node should be merged with its children"""

        if not children:
            return False

        attrs = node.attrib
        class_name = attrs.get("class", "")

        # don't merge: semantic types: it's important to stay
        if class_name in self.SEMANTIC_TYPES:
            return False

        # merge: if this is a container with a single child & the parent (node) has no semantic info
        if len(children) == 1 and class_name in self.CONTAINER_TYPES:
            parent_has_info = (
                bool(attrs.get("text", "").strip())
                or bool(attrs.get("content-desc", "").strip())
                or self._is_interactive_xml(node)
            )
            if not parent_has_info:
                return True

        # merge: consecutive text elements
        if class_name == "android.widget.TextView" and len(children) == 1:
            if children[0].tag == "android.widget.TextView":
                return True

        return False

    def _should_merge_proto_with_children(
        self, node: AndroidAccessibilityNodeInfo, children: List[View]
    ) -> bool:
        """Determine if protobuf node should be merged with its children"""

        if not children:
            return False

        class_name = node.class_name or ""

        # don't merge: semantic types: it's important to stay
        if class_name in self.SEMANTIC_TYPES:
            return False

        # merge: if this is a container with a single child & the parent (node) has no semantic info
        if len(children) == 1 and class_name in self.CONTAINER_TYPES:
            parent_has_info = (
                bool((node.text or "").strip())
                or bool((node.content_description or "").strip())
                or self._is_interactive_proto(node)
            )
            if not parent_has_info:
                return True

        # merge: consecutive text elements
        if class_name == "android.widget.TextView" and len(children) == 1:
            if children[0].tag == "android.widget.TextView":
                return True

        return False

    def _merge_xml_node_with_children(
        self, node: ET.Element, children: List[View]
    ) -> View:
        """Merge xml parent node with its children"""

        # single child
        if len(children) == 1:
            child = children[0]
            attrs = node.attrib

            # merge text: parent text takes precedence
            parent_text = attrs.get("text", "").strip()
            if parent_text and not child.text:
                child.text = parent_text

            # merge content description
            parent_desc = attrs.get("content-desc", "").strip()
            if parent_desc and not child.content_desc:
                child.content_desc = parent_desc

            # merge resource ID
            parent_id = attrs.get("resource-id", "").strip()
            if parent_id and not child.resource_id:
                child.resource_id = parent_id

            # merge interactive properties
            child.clickable = child.clickable or attrs.get("clickable") == "true"
            child.scrollable = child.scrollable or attrs.get("scrollable") == "true"
            child.checkable = child.checkable or attrs.get("checkable") == "true"

            return child

        # multiple children: create container
        view = self._create_view_from_xml(node)
        view.children = children
        return view

    def _merge_proto_node_with_children(
        self, node: AndroidAccessibilityNodeInfo, children: List[View]
    ) -> View:
        """Merge protobuf parent node with its children"""

        # single child
        if len(children) == 1:
            child = children[0]

            # merge text: parent text takes precedence
            parent_text = (node.text or "").strip()
            if parent_text and not child.text:
                child.text = parent_text

            # merge content description
            parent_desc = (node.content_description or "").strip()
            if parent_desc and not child.content_desc:
                child.content_desc = parent_desc

            # merge resource ID
            parent_id = (node.view_id_resource_name or "").strip()
            if parent_id and not child.resource_id:
                child.resource_id = parent_id

            # merge interactive properties
            child.clickable = child.clickable or node.is_clickable
            child.scrollable = child.scrollable or node.is_scrollable
            child.checkable = child.checkable or node.is_checkable

            return child

        # multiple children: create container
        view = self._create_view_from_proto(node)
        view.children = children
        return view

    def _is_leaf_filtered_xml(self, node: ET.Element) -> bool:
        """Determine if xml leaf node should be filtered"""

        attrs = node.attrib

        if not self._is_visible_xml(node):
            return True

        has_text = bool(attrs.get("text", "").strip())
        has_content_desc = bool(attrs.get("content-desc", "").strip())
        has_resource_id = bool(attrs.get("resource-id", "").strip())
        is_interactive = self._is_interactive_xml(node)

        class_name = attrs.get("class", "")
        is_image = "Image" in class_name

        # filter: empty element
        if is_image and not (has_content_desc or has_resource_id or is_interactive):
            return True

        # keep: if has any semantic value
        return not (has_text or has_content_desc or has_resource_id or is_interactive)

    def _is_leaf_filtered_proto(self, node: AndroidAccessibilityNodeInfo) -> bool:
        """Determine if protobuf leaf node should be filtered"""

        if not self._is_visible_proto(node):
            return True

        has_text = bool((node.text or "").strip())
        has_content_desc = bool((node.content_description or "").strip())
        has_resource_id = bool((node.view_id_resource_name or "").strip())
        is_interactive = self._is_interactive_proto(node)

        class_name = node.class_name or ""
        is_image = "Image" in class_name

        # filter: empty element
        if is_image and not (has_content_desc or has_resource_id or is_interactive):
            return True

        # keep: if has any semantic value
        return not (has_text or has_content_desc or has_resource_id or is_interactive)

    def _create_view_from_xml(self, node: ET.Element) -> View:
        """Create a View object from xml node"""

        attrs = node.attrib

        # shorten class name
        class_name = attrs.get("class", "")
        tag = class_name.split(".")[-1] if "." in class_name else class_name

        # map to SEMANTIC TAG
        semantic_tag = self.TAG_MAPPING.get(tag, tag)

        unique_id = f"ui_{self.unique_id_counter}"
        self.unique_id_counter += 1

        bounds = attrs.get("bounds", "")

        self.id_to_bounds[unique_id] = bounds

        return View(
            tag=semantic_tag,
            unique_id=unique_id,
            text=self._clean_text(attrs.get("text", "")),
            resource_id=attrs.get("resource-id", "").strip(),
            content_desc=self._clean_text(attrs.get("content-desc", "")),
            clickable=attrs.get("clickable") == "true",
            scrollable=attrs.get("scrollable") == "true",
            checkable=attrs.get("checkable") == "true",
            bounds=attrs.get("bounds", ""),
        )

    def _create_view_from_proto(self, node: AndroidAccessibilityNodeInfo) -> View:
        """Create a View object from protobuf node"""

        class_name = node.class_name or ""
        tag = class_name.split(".")[-1] if "." in class_name else class_name
        semantic_tag = self.TAG_MAPPING.get(tag, tag)

        unique_id = f"ui_{self.unique_id_counter}"
        self.unique_id_counter += 1

        bounds = f"[{node.bounds_in_screen.left},{node.bounds_in_screen.top}][{node.bounds_in_screen.right},{node.bounds_in_screen.bottom}]"

        self.id_to_bounds[unique_id] = bounds

        return View(
            tag=semantic_tag,
            unique_id=unique_id,
            text=self._clean_text(node.text or ""),
            resource_id=(node.view_id_resource_name or "").strip(),
            content_desc=self._clean_text(node.content_description or ""),
            clickable=node.is_clickable,
            scrollable=node.is_scrollable,
            checkable=node.is_checkable,
            bounds=bounds,
        )

    def _is_visible_xml(self, node: ET.Element) -> bool:
        """Check if xml node is visible"""

        bounds = node.attrib.get("bounds", "")
        if bounds:
            try:
                parts = bounds.replace("[", "").replace("]", ",").split(",")
                x1, y1, x2, y2 = map(int, parts[:4])

                # zero-sized
                if x1 == x2 or y1 == y2:
                    return False
            except:
                pass

        return True

    def _is_visible_proto(self, node: AndroidAccessibilityNodeInfo) -> bool:
        """Check if protobuf node is visible"""

        # zero-sized
        rect = node.bounds_in_screen
        if rect.left == rect.right or rect.top == rect.bottom:
            return False

        return node.is_visible_to_user and node.is_enabaled

    def _is_interactive_xml(self, node: ET.Element) -> bool:
        """Check if xml node is interactive"""

        attrs = node.attrib
        return (
            attrs.get("clickable") == "true"
            or attrs.get("long-clickable") == "true"
            or attrs.get("scrollable") == "true"
            or attrs.get("checkable") == "true"
            or attrs.get("focusable") == "true"
        )

    def _is_interactive_proto(self, node: AndroidAccessibilityNodeInfo) -> bool:
        """Check if protobuf node is interactive"""

        return (
            node.is_clickable
            or node.is_long_clickable
            or node.is_scrollable
            or node.is_checkable
            or node.is_focusable
        )

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = " ".join(text.split())

        text = text.replace("\u200d", "")  # Zero-width joiner
        text = text.replace("\ufe0f", "")  # Variation selector
        text = text.replace("\u200b", "")  # Zero-width space

        text = re.sub(r"([\U0001F300-\U0001F9FF])\1+", r"\1", text)

        # Limit total consecutive emojis to 3
        def limit_emojis(match):
            emojis = match.group(0)
            # Keep max 3 different emojis in a row
            return "".join(list(emojis)[:3])

        text = re.sub(r"[\U0001F300-\U0001F9FF]{4,}", limit_emojis, text)

        text = " ".join(text.split())

        return text.strip()

    def _count_nodes(self, node: ET.Element) -> int:
        """Count total nodes in tree"""
        return 1 + sum(self._count_nodes(child) for child in node)

    def _count_views(self, views: List[View]) -> int:
        """Count total views in compressed tree"""
        count = len(views)
        for view in views:
            count += self._count_views(view.children)
        return count

    def get_compression_stats(self) -> Dict[str, any]:
        """Get compression statistics"""
        reduction = 0
        if self.original_nodes > 0:
            reduction = (1 - self.compressed_nodes / self.original_nodes) * 100

        return {
            "original_nodes": self.original_nodes,
            "compressed_nodes": self.compressed_nodes,
            "reduction_percent": round(reduction, 1),
        }


def extract_and_parse_ui_tree(
    compressor: UITreeCompressor,
) -> Tuple[str, Dict[str, str]]:
    """Extract UI tree from device and compress it using UITreeCompressor.

    Args:
        compressor: UITreeCompressor instance to use for compression

    Returns:
        tuple: (compressed_tree_string, id_to_bounds_dict)

    Raises:
        ValueError: If XML extraction fails
        subprocess.CalledProcessError: If adb command fails
    """
    result = subprocess.run(
        ["adb", "exec-out", "uiautomator", "dump", "/dev/tty"],
        capture_output=True,
        text=True,
        check=True,
    )
    xml_str = result.stdout

    # Find XML start
    xml_start = xml_str.find("<?xml")
    if xml_start == -1:
        raise ValueError("No XML found in uiautomator output")

    # Find XML end - look for the closing hierarchy tag
    xml_end = xml_str.find("</hierarchy>", xml_start)
    if xml_end == -1:
        raise ValueError("No closing </hierarchy> tag found")

    # Extract only the valid XML portion
    xml_str = xml_str[xml_start : xml_end + len("</hierarchy>")]

    # Use the compressor to compress the XML string
    compressed_tree, id_to_bounds = compressor.compress_xml_string(xml_str)

    return compressed_tree, id_to_bounds


class GroqAgent(EnvironmentInteractingAgent):
    def __init__(
        self,
        env,
        api_key,
        model_name,
        name="groq",
    ):
        super().__init__(env=env, name=name)

        # init client
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

        self.prev_actions = []
        self.compressor = UITreeCompressor()

    def step(self, goal) -> AgentInteractionResult:
        """Execute one step: get UI, generate action, execute it"""

        compressed_tree, id_to_bounds = extract_and_parse_ui_tree(self.compressor)

        prompt = self.create_prompt(goal, compressed_tree)

        response = self.generate_action(prompt)

        action_json = self.parse_json_response(response)

        print(f"Generated action: {action_json}")

        if action_json is None:
            raise ValueError("Model did not return valid JSON")

        # Convert element_id to coordinates for AndroidWorld execution
        action_for_execution = self.convert_id_to_coords(action_json, id_to_bounds)

        # Create JSONAction for AndroidWorld
        action = json_action.JSONAction(**action_for_execution)

        # Execute action if not done
        if action.action_type != "done":
            self.env.execute_action(action)

        # Store original action (with element_id) for previous actions
        self.prev_actions.append(action_json)

        return AgentInteractionResult(
            done=(action.action_type == "done"),
            data={"action": action, "goal": goal},
        )

    def create_prompt(self, goal: str, ui_elements: str) -> list:
        """Create prompt matching training data format"""

        # Format previous actions as bullet points (matching training)
        if self.prev_actions:
            prev_actions_str = "\n".join(
                [f"- {self.action_to_string(a)}" for a in self.prev_actions[-3:]]
            )
        else:
            prev_actions_str = "None"

        # Match training data structure
        messages = [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are an Android UI navigation agent.",
                        "Your task is to select the next action required to accomplish the given goal.",
                        "You will be given:",
                        "- the goal",
                        "- previous actions already executed",
                        "- the current UI screen representation",
                        "- the action schema (Pydantic)",
                        "",
                        "Choose the best next action that moves toward completing the goal.",
                        "Only output a valid JSON object that follows the Pydantic schema.",
                        "Do not include explanations, comments, or extra text.",
                        "",
                        "## Pydantic Details:",
                        json.dumps(action_space_schema, indent=2),
                        "",
                        "## UI Tree Format:",
                        "The Current UI is represented as a hierarchical tree structure:",
                        "- Indentation shows parent-child relationships",
                        "- Format: [element_id] element_type (properties)",
                        "- Properties include:",
                        "  - text: visible text content",
                        "  - desc: content description/accessibility label",
                        "  - id: Android resource ID",
                        "  - [clickable]: element is interactive",
                        "  - [scrollable]: element can be scrolled",
                        "- To interact with an element, use its element_id (e.g., 'ui_42')",
                        "- Elements without [clickable] cannot be clicked",
                        "",
                    ]
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "## Goal:",
                        goal,
                        "",
                        "## Previous Actions:",
                        prev_actions_str,
                        "",
                        "## Current UI:",
                        ui_elements,
                        "",
                        "## Next Action:",
                        "```json",
                    ]
                ),
            },
        ]

        return messages

    def parse_json_response(self, response: str) -> Optional[dict]:
        """Parse JSON from model response, handling markdown code blocks"""

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            response = json_match.group(1)

        # Also try without markdown
        json_match = re.search(r"\{.*?\}", response, re.DOTALL)
        if json_match:
            response = json_match.group(0)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response was: {response[:200]}")
            return None

    def convert_id_to_coords(self, action: dict, id_to_bounds: Dict[str, str]) -> dict:
        """Convert element_id to coordinates for AndroidWorld execution"""

        action_type = action.get("action_type")

        # Actions that use element_id need conversion to coordinates
        if action_type in ["click", "long_press"]:
            element_id = action.get("element_id")

            if element_id and element_id in id_to_bounds:
                bounds_str = id_to_bounds[element_id]

                try:
                    # Parse bounds like "[100,200][300,400]"
                    parts = bounds_str.replace("[", "").replace("]", ",").split(",")
                    x1, y1, x2, y2 = map(int, parts[:4])

                    # Use center of bounds
                    x = (x1 + x2) // 2
                    y = (y1 + y2) // 2

                    return {"action_type": action_type, "x": x, "y": y}

                except Exception as e:
                    print(f"Error parsing bounds for {element_id}: {e}")
                    # Fallback: return original action
                    return action
            else:
                print(f"Element ID '{element_id}' not found in bounds mapping")
                # Fallback: return original action
                return action

        # Other actions don't need conversion (scroll, input_text, etc.)
        return action

    def action_to_string(self, action: dict) -> str:
        """Convert action dict to string representation (matching training)"""

        action_type = action.get("action_type")

        if action_type == "click":
            return f"click({action.get('element_id', '?')})"

        elif action_type == "long_press":
            return f"long_press({action.get('element_id', '?')})"

        elif action_type == "scroll":
            direction = action.get("direction", "?")
            return f"scroll({direction})"

        elif action_type == "input_text":
            text = action.get("text", "")
            # Escape and shorten text if too long
            text_display = text[:30] + "..." if len(text) > 30 else text
            return f"input_text('{text_display}')"

        elif action_type == "open_app":
            return f"open_app({action.get('app_name', '?')})"

        elif action_type in ["navigate_home", "navigate_back", "wait", "done"]:
            return f"{action_type}"

        else:
            return json.dumps(action)

    def generate_action(self, messages: list) -> str:
        """Generate action using Groq API"""

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,  # Deterministic for evaluation
            max_tokens=512,
            response_format={"type": "json_object"},  # Force JSON output
        )

        return completion.choices[0].message.content
