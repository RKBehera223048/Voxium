---
name: Ancient Intelligence
colors:
  surface: '#fff9ed'
  surface-dim: '#e2dabf'
  surface-bright: '#fff9ed'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#fcf3d8'
  surface-container: '#f7eed2'
  surface-container-high: '#f1e8cd'
  surface-container-highest: '#ebe2c8'
  on-surface: '#1f1c0b'
  on-surface-variant: '#58413f'
  inverse-surface: '#35301e'
  inverse-on-surface: '#faf0d5'
  outline: '#8c716e'
  outline-variant: '#e0bfbc'
  surface-tint: '#ad302f'
  primary: '#840f16'
  on-primary: '#ffffff'
  primary-container: '#a52a2a'
  on-primary-container: '#ffc0bb'
  inverse-primary: '#ffb3ad'
  secondary: '#4e6078'
  on-secondary: '#ffffff'
  secondary-container: '#cee1fe'
  on-secondary-container: '#52647c'
  tertiary: '#533e00'
  on-tertiary: '#ffffff'
  tertiary-container: '#6f5400'
  on-tertiary-container: '#fec72d'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#ffdad7'
  primary-fixed-dim: '#ffb3ad'
  on-primary-fixed: '#410004'
  on-primary-fixed-variant: '#8c171b'
  secondary-fixed: '#d2e4ff'
  secondary-fixed-dim: '#b5c8e4'
  on-secondary-fixed: '#081c32'
  on-secondary-fixed-variant: '#36485f'
  tertiary-fixed: '#ffdf98'
  tertiary-fixed-dim: '#f5bf22'
  on-tertiary-fixed: '#251a00'
  on-tertiary-fixed-variant: '#5a4300'
  background: '#fff9ed'
  on-background: '#1f1c0b'
  surface-variant: '#ebe2c8'
typography:
  display-lg:
    fontFamily: Literata
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Literata
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  headline-md:
    fontFamily: Literata
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Work Sans
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Work Sans
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-sm:
    fontFamily: Work Sans
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 8px
  container-margin: 24px
  gutter: 16px
  panel-padding: 24px
---

## Brand & Style

The design system is a synthesis of traditional Madhubani tribal art and futuristic AI architecture. It positions the "Always-On" assistant not as a cold machine, but as a living, breathing digital companion—a keeper of stories and a modern-day scribe.

The style is **Tactile / Folk-Futurism**. It utilizes hand-drawn textures, intricate geometric borders (Bharni and Katchni styles), and organic line work to ground the high-tech functionality in human history. The interface should feel like an illuminated manuscript on parchment, where every interaction feels deliberate and personal.

**Key Visual Pillars:**
- **Organic Geometry:** Borders are never perfectly straight; they carry the "vibration" of a hand-drawn line.
- **Mythological Motifs:** Use of stylized suns for system status, fish/birds for data flow, and lotus patterns for "knowledge centers."
- **Storytelling Layouts:** Information is presented in tiered panels that mimic traditional folk murals, using line work to connect disparate data points.

## Colors

The palette is derived from natural pigments used in Mithila painting. The background is a textured, warm cream that reduces digital eye strain and provides a soft "paper" feel.

- **Base (Cream):** The primary canvas. Used for all background surfaces to maintain the parchment aesthetic.
- **Primary (Madhubani Red):** Reserved for high-priority actions, primary buttons, and critical status indicators.
- **Secondary (Indigo Blue):** Used for navigation elements, secondary buttons, and links. It represents depth and wisdom.
- **Tertiary (Mustard Yellow):** An accent for highlighting, active states, and decorative flourishes within motifs.
- **Ink (Charcoal Black):** Used for all typography and the signature hand-drawn borders. It mimics the soot-based ink of traditional tribal art.

## Typography

This design system uses a dual-font approach to balance editorial elegance with functional clarity.

- **The Serif (Literata):** Used for headlines and storytelling elements. Its calligraphic roots and high legibility echo the strokes of Devanagari script, providing a bridge to the Madhubani theme.
- **The Sans-Serif (Work Sans):** Used for all UI controls, body text, and data-heavy tables. Its neutral, professional tone ensures that the intricate art style doesn't compromise usability or readability.

**Text Styling:**
Use sentence case for most UI elements to maintain a friendly, personal tone. All-caps should be reserved for small labels and utility text using **Work Sans** with increased letter-spacing.

## Layout & Spacing

The layout follows a **Fluid Grid** model with a "Panel-in-Panel" hierarchy. Unlike modern flat designs, sections are defined by intricate decorative borders rather than just whitespace.

- **Desktop:** A 12-column grid with wide margins (48px+) to allow the decorative border art to "breathe" around the edges.
- **Mobile:** A single-column flow where decorative borders transition into simple 1px ink lines to maximize screen real estate.
- **Spacing Rhythm:** Based on an 8px scale. However, internal padding in cards and containers should be generous (24px+) to prevent the complex border textures from feeling cluttered.

## Elevation & Depth

In this design system, depth is achieved through **Tonal Layers** and line weight rather than shadows. 

- **Surface Levels:** The base level is the darkest cream. Active panels or cards use a slightly lighter cream (#FAF7ED) to "lift" them visually.
- **Line Hierarchy:** Depth is communicated by the complexity of the border. Global containers have heavy, double-lined borders with geometric patterns (Katchni style). Interactive elements have single, slightly irregular "hand-drawn" outlines.
- **Zero Shadows:** Shadows are avoided to stay true to the 2D nature of traditional Indian folk art. Instead, use a subtle inner stroke or a tint of Mustard Yellow to indicate focus or elevation.

## Shapes

The shape language is "Soft-Organic." While the layout is structured, the corners and edges should never feel mathematically perfect.

- **Border Corners:** Use a subtle 0.25rem (4px) radius. 
- **The "Wobble":** Every border should have a slight SVG displacement map or hand-drawn texture to ensure it doesn't look like a standard CSS border.
- **Motif Enclosures:** Circular elements (like avatars or status icons) should use a "Sun" or "Lotus" petal frame rather than a simple circle.

## Components

### Buttons
- **Primary:** Solid Madhubani Red fill with an inner "stitched" white line. Rounded-sm.
- **Secondary:** Transparent with an Indigo Blue hand-drawn border.
- **Tertiary/Ghost:** Text-only in Charcoal Black with a small "leaf" motif that appears on hover.

### Cards
- Background: Lighter cream (#FAF7ED).
- Border: 1px Charcoal Black with an outer decorative "frieze" pattern on the top edge.
- Interaction: On hover, the border thickens slightly or shifts to Mustard Yellow.

### Input Fields
- Styled as "underlined" parchment lines. The label sits above in Work Sans Medium.
- Focus state: The underline turns into a thin decorative vine or geometric pattern.

### Icons (Tribal Glyphs)
- Icons must not be standard material/linear icons. They should be custom-drawn in Charcoal Black, using thick-and-thin strokes consistent with Madhubani nibs.
- *Examples:* Use a stylized "Eye" for view, a "Sun" for settings, and a "Brahmi-style Bird" for sending messages.

### Status Indicators
- Instead of simple dots, use small suns (active), moons (idle), or seeds (processing).