---
name: Obsidian Engineering Console
colors:
  surface: '#10131a'
  surface-dim: '#10131a'
  surface-bright: '#363941'
  surface-container-lowest: '#0b0e15'
  surface-container-low: '#191b23'
  surface-container: '#1d2027'
  surface-container-high: '#272a31'
  surface-container-highest: '#32353c'
  on-surface: '#e1e2ec'
  on-surface-variant: '#c2c6d6'
  inverse-surface: '#e1e2ec'
  inverse-on-surface: '#2e3038'
  outline: '#8c909f'
  outline-variant: '#424754'
  surface-tint: '#adc6ff'
  primary: '#adc6ff'
  on-primary: '#002e6a'
  primary-container: '#4d8eff'
  on-primary-container: '#00285d'
  inverse-primary: '#005ac2'
  secondary: '#b7c8e1'
  on-secondary: '#213145'
  secondary-container: '#3a4a5f'
  on-secondary-container: '#a9bad3'
  tertiary: '#ffb786'
  on-tertiary: '#502400'
  tertiary-container: '#df7412'
  on-tertiary-container: '#461f00'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a42'
  on-primary-fixed-variant: '#004395'
  secondary-fixed: '#d3e4fe'
  secondary-fixed-dim: '#b7c8e1'
  on-secondary-fixed: '#0b1c30'
  on-secondary-fixed-variant: '#38485d'
  tertiary-fixed: '#ffdcc6'
  tertiary-fixed-dim: '#ffb786'
  on-tertiary-fixed: '#311400'
  on-tertiary-fixed-variant: '#723600'
  background: '#10131a'
  on-background: '#e1e2ec'
  surface-variant: '#32353c'
typography:
  display-lg:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '600'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: 32px
    letterSpacing: -0.01em
  body-base:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Geist
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  mono-label:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
  mono-data:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  gutter: 16px
  margin: 24px
---

## Brand & Style

This design system is engineered for high-density information environments, research terminals, and mission-critical engineering interfaces. It rejects the ephemeral trends of "AI-aesthetic" glows in favor of institutional permanence and technical rigor.

The visual style is **Mechanical Minimalism**. It draws inspiration from aerospace telemetry and financial terminals, prioritizing clarity, structural integrity, and low cognitive load during prolonged use. The aesthetic is defined by "Midnight Obsidian" depth, high-contrast legibility, and a strict adherence to a logic-driven layout. Every element must feel intentional, durable, and precise.

## Colors

The palette is anchored in a deep-space spectrum to reduce eye strain and establish a hierarchy of information.

- **Background (Midnight Obsidian):** Used for the primary canvas to provide maximum depth.
- **Surface (Steel Slate):** Used for primary containers, sidebars, and panels. This creates a clear structural distinction without relying on shadows.
- **Primary Accent (Command Indigo):** Reserved for primary actions, active states, and critical focus points. It provides a sharp, authoritative "ping" against the dark background.
- **Neutral Scale:** Utilizes a range of cool greys to differentiate between labels, secondary text, and inactive UI elements.
- **Status Colors:** Use standard semantic reds (Error), ambers (Warning), and emeralds (Success), but desaturated slightly to match the professional tone of the system.

## Typography

The system utilizes a dual-font strategy to separate interface navigation from technical data analysis.

- **Interface Sans (Geist):** Used for all navigational elements, headers, and standard body text. Its geometric precision ensures legibility at small sizes while maintaining a modern, engineering-first character.
- **Technical Mono (JetBrains Mono):** Used exclusively for data points, code blocks, coordinates, timestamps, and status labels. The monospaced nature allows for vertical alignment of data columns, essential for rapid scanning of telemetry or research results.
- **Weight Usage:** Avoid "Extra Bold" weights. Use Medium (500) for emphasis and Regular (400) for standard data to maintain a sophisticated, understated profile.

## Layout & Spacing

This design system employs a strict **4px/8px incremental grid**. Precision is the primary visual cue for quality; all components must align to this base unit.

- **Modular Panels:** Layouts should be constructed using fixed or semi-fluid panels separated by 1px dividers rather than large gaps of white space.
- **Density:** High information density is encouraged. Padding within containers should be tight (8px or 12px) to maximize the amount of visible data on screen.
- **Grid System:** A 12-column grid is used for dashboard views, while a 1px vertical "rail" system is used for narrow sidebars and navigation panels.
- **Breakpoints:** 
    - Mobile: Full-width stacked panels.
    - Tablet: Multi-panel view with collapsible sidebar.
    - Desktop (1440px+): Full multi-column telemetry view.

## Elevation & Depth

Depth in this system is achieved through **Tonal Layering** and **Structural Outlines** rather than traditional shadows.

- **The 1px Rule:** All containers and interactive elements use a 1px solid border. The border color should be slightly lighter than the surface it sits on (e.g., `#334155` on a `#1E293B` surface).
- **Z-Axis Hierarchy:**
    - **Level 0 (Background):** `#0F1115` - The base layer.
    - **Level 1 (Panels):** `#1E293B` - Main content areas.
    - **Level 2 (Active/Hover):** A slightly lighter slate or a 1px Indigo border to indicate focus.
- **Dividers:** Use hairline 1px dividers (`#334155`) to separate logical groups within a single panel.
- **Shadows:** Only used for temporary floating elements (Modals/Popovers). When used, they must be sharp, neutral, and low-spread to maintain the "flat" engineering feel.

## Shapes

The shape language is **Soft-Industrial**. While the overall feel is rigid and structured, subtle rounding prevents the UI from feeling aggressive or dated.

- **Base Radius:** 4px (Soft) for buttons, input fields, and small containers.
- **Container Radius:** 8px for large dashboard panels or modals.
- **Sharpness:** Inner elements (like segments in a toggle) should have a smaller radius than their parent container to maintain visual nesting logic.
- **Interaction States:** No change in radius on hover; maintain structural consistency.

## Components

- **Buttons:** Solid Indigo (#3B82F6) with white text for Primary. Ghost buttons with 1px Slate borders for Secondary. Label text must be all-caps for utility actions.
- **Inputs:** Darker background than the surface, 1px border. Focus state is a 1px Indigo border with no outer glow.
- **Data Tables:** The core of the system. No alternating row colors. Use 1px horizontal dividers. Header cells use JetBrains Mono in a dim grey.
- **Chips/Status:** Small, rectangular with a 2px radius. Use a "dot" indicator for status (e.g., a green dot for 'Active') rather than coloring the entire chip background.
- **Telemetry Cards:** Solid surfaces with no shadow. Header of the card should be separated by a 1px divider and contain a "Technical Mono" label.
- **Scrollbars:** Custom, slim (6px), dark slate tracks with slightly lighter slate thumbs to blend into the interface.