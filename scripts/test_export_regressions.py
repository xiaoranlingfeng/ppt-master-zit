"""Regression tests for bugs found while generating a real deck (GPU-free).

Each test locks in a failure that actually shipped once and was caught only by
rendering the exported PPTX, not by the quality gate.
"""

import os
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DML_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PML_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
D = f"{{{DML_NS}}}"
P = f"{{{PML_NS}}}"


def _paragraph_pr_xml(*, algn: str, body_xml: str = "") -> str:
    from svg_to_pptx.drawingml.elements import _paragraph_pr_xml as impl

    return impl(algn=algn, font_size=24, body_xml=body_xml)


class TestInheritedBullets(unittest.TestCase):
    """Ordinary prose must not inherit the slide master's bodyStyle bullet."""

    def test_plain_paragraph_disables_inherited_bullet(self):
        xml = _paragraph_pr_xml(algn="l")
        self.assertIn("<a:buNone", xml)
        self.assertIn('marL="0"', xml)
        self.assertIn('indent="0"', xml)

    def test_leading_spacer_children_precede_bu_none(self):
        # buNone must follow spcBef/spcAft, which DrawingML requires in order.
        xml = _paragraph_pr_xml(
            algn="ctr",
            body_xml='<a:lnSpc><a:spcPts val="2000"/></a:lnSpc>',
        )
        self.assertLess(xml.index("lnSpc"), xml.index("buNone"))

    def test_explicit_bullet_is_preserved(self):
        from svg_to_pptx.drawingml.elements import _paragraph_pr_xml as impl

        xml = impl(
            algn="l",
            font_size=24,
            bullet={"char": "•", "margin_px": 20.0},
        )
        self.assertIn('buChar char="•"', xml)
        self.assertNotIn("buNone", xml)
        self.assertIn("marL=", xml)


class TestPlaceholderBulletScope(unittest.TestCase):
    """Every placeholder kind must be bullet-free, not just body/subtitle."""

    def _item(self, placeholder: str):
        from svg_to_pptx.pptx_package.template_structure import TemplateElementSpec

        return TemplateElementSpec(
            element_id="slot",
            tag="g",
            layer=None,
            placeholder=placeholder,
            placeholder_bounds=(0, 0, 100, 100),
            placeholder_idx=None,
            placeholder_binding=None,
            placeholder_carrier_tag=None,
            is_background=False,
            order=0,
        )

    def _shape_with_two_paragraphs(self):
        shape = ET.Element(f"{P}sp")
        tx_body = ET.SubElement(shape, f"{P}txBody")
        for _ in range(2):
            ET.SubElement(tx_body, f"{D}p")
        return shape

    def test_object_slot_paragraphs_are_bullet_free(self):
        from svg_to_pptx.pptx_package.builder import (
            _set_placeholder_no_inherited_bullets,
        )

        shape = self._shape_with_two_paragraphs()
        _set_placeholder_no_inherited_bullets(shape, self._item("object"))
        for paragraph in shape.iter(f"{D}p"):
            props = paragraph.find(f"{D}pPr")
            self.assertIsNotNone(props, "paragraph gained no pPr")
            self.assertIsNotNone(props.find(f"{D}buNone"))
            self.assertEqual(props.get("marL"), "0")

    def test_title_placeholder_paragraphs_are_bullet_free(self):
        from svg_to_pptx.pptx_package.builder import (
            _set_placeholder_no_inherited_bullets,
        )

        shape = self._shape_with_two_paragraphs()
        _set_placeholder_no_inherited_bullets(shape, self._item("title"))
        for paragraph in shape.iter(f"{D}p"):
            self.assertIsNotNone(paragraph.find(f"{D}pPr/{D}buNone"))


class TestMasterStyleBullets(unittest.TestCase):
    """The generated Master text style must not define a default bullet."""

    def _style(self):
        style = ET.Element(f"{P}bodyStyle")
        for level in range(1, 4):
            props = ET.SubElement(style, f"{D}lvl{level}pPr")
            props.set("marL", "342900")
            props.set("indent", "-342900")
            ET.SubElement(props, f"{D}buFont").set("typeface", "Arial")
            ET.SubElement(props, f"{D}buChar").set("char", "•")
            ET.SubElement(props, f"{D}defRPr").set("sz", "1800")
        return style

    def test_default_bullets_are_cleared(self):
        from svg_to_pptx.drawingml.theme_fonts import _clear_default_bullets

        style = self._style()
        _clear_default_bullets(style)
        for props in style.iter(f"{D}lvl1pPr"):
            self.assertIsNone(props.find(f"{D}buChar"))
            self.assertIsNotNone(props.find(f"{D}buNone"))
            self.assertNotIn("marL", props.attrib)
            self.assertNotIn("indent", props.attrib)


class TestMossTemplateContract(unittest.TestCase):
    """MOSS-Image SFT must use ModelSamplingAuraFlow, not ModelSamplingSD3."""

    def test_sampling_node_is_aura_flow(self):
        from image_backends import backend_comfyui as bc

        graph = bc._load_template(bc._COMFY_TEMPLATE_DIR / "moss_image_sft_api.json")
        classes = {node["class_type"] for node in graph.values()}
        self.assertIn("ModelSamplingAuraFlow", classes)
        self.assertNotIn("ModelSamplingSD3", classes)


class TestDimensionAlignment(unittest.TestCase):
    """MOSS needs 16px alignment and a 64px floor; Qwen/Z-Image need 32px."""

    def test_model_specific_rules(self):
        from image_backends import backend_comfyui as bc

        self.assertEqual(bc._apply_dimension_rule(1030, 1030, "moss"), (1024, 1024))
        self.assertEqual(bc._apply_dimension_rule(1030, 1030, "qwen"), (1024, 1024))
        self.assertEqual(bc._apply_dimension_rule(30, 30, "moss"), (64, 64))
        self.assertEqual(bc._apply_dimension_rule(1030, 1030, None), (1024, 1024))


if __name__ == "__main__":
    unittest.main()
