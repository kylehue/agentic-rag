import { describe, expect, it } from "vitest";
import { mount } from "../../tests/mount";
import Button from "./button.vue";

describe("Button", () => {
    it("renders the default variant", () => {
        const wrapper = mount(Button, {
            props: { class: "" },
            slots: { default: "Go" },
        });
        expect(wrapper.text()).toBe("Go");
        expect(wrapper.classes()).toContain("bg-primary");
    });

    it("applies variant and size classes", () => {
        const destructive = mount(Button, {
            props: { variant: "destructive", size: "icon" },
        });
        expect(destructive.classes()).toContain("bg-destructive");
        expect(destructive.classes()).toContain("w-9");

        const outline = mount(Button, { props: { variant: "outline" } });
        expect(outline.classes()).toContain("border");
    });

    it("sets the disabled attribute", () => {
        const wrapper = mount(Button, { props: { disabled: true } });
        expect(wrapper.attributes("disabled")).toBeDefined();
    });

    it("emits clicks", async () => {
        const wrapper = mount(Button);
        await wrapper.trigger("click");
        expect(wrapper.emitted("click")).toHaveLength(1);
    });
});
